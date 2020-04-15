import os

import numpy as np
import pickle

from sklearn.model_selection import train_test_split

import tensorflow as tf


# TODO: Simplify for understanding and proprietary reasons (single model with hedge and loss layers)
# TODO: Use data feed pipline as reccommended for TF 2.0
# TODO: Use feature columns



##===============================================================
## SETTINGS
##===============================================================

##--- GPU USAGE
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

##--- LOADING DATA
fpath = './results/sims/vanilla_sim_6SPEC+D90_1dim.pkl'

##--- BID/ASK
ba_spread = 0.01

##--- TRAINING/TESTING
feat_def = ['SPOT', 'BIDASK_SPREAD', 'CALL_TTM']
train_pct = 0.8
random_seed = 42

##--- NETWORK PARAMETER
units = [16, 8, 1]
activations = ['tanh', 'tanh', 'linear']

##--- LOSS AND OPTIMIZATION PARAMETER
loss_type = 'expected_shortfall_10pct' # allowed are: expected_shortfall_10pct, mean_squared_error
learning_rate = 0.001
batch_size = 128
epochs = 500



##===============================================================
## HELPER FUNCTIONS
##===============================================================

@tf.function
def expected_shortfall_10pct(y_true, y_pred):
    pnl = tf.keras.backend.reshape(y_pred - y_true, (-1, ))
    pnl = tf.multiply(pnl, -1.0)

    n_pct = tf.keras.backend.round(tf.multiply(
        tf.keras.backend.cast(tf.keras.backend.shape(pnl)[0], tf.float32)  , 0.1))

    pnl_past_cutoff = tf.nn.top_k(pnl, tf.keras.backend.cast(n_pct, tf.int32))[0]
    return tf.reduce_mean(pnl_past_cutoff)


##===============================================================
## CLASSES
##===============================================================

class Hedger(tf.keras.Model):

    def __init__(
        self,
        n_feat,
        n_time,
        model_param):

        # call parent class
        super(Hedger, self).__init__()

        # init hedge model
        self.hedge_model = HedgeModel(n_feat, n_time, model_param['units'], model_param['activations'])

        # init cashflow model
        self.cf_model = CashflowModel()

    def call(self, inputs, training=False):
        
        f_hedge = inputs[0]
        hedge_qty = self.hedge_model.call(f_hedge, training=training)

        f_loss = [hedge_qty] + inputs[1:]
        loss_cf = self.cf_model.call(f_loss, training=training)

        return loss_cf



class HedgeModel(tf.keras.Model):

    def __init__(
        self,
        n_feat,
        n_time,
        units,
        activations):

        super(HedgeModel, self).__init__()

        # Checks
        if units[-1] != 1:
            raise ValueError('Last unit must be 1, because it specifies the hedge quantity for the underlying stock.')

        # Members
        self.n_feat = n_feat
        self.n_times = n_time
        self.n_units = len(units)
        
        # Dense layer
        self.lay_dnn_0 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units[0], activation=activations[0]))

        for k in range(1, len(units)-1):
            setattr(
                self,
                'lay_dnn_' + str(k), 
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units[k], activation=activations[k]))
            )

        self.lay_hedge_qty = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units[-1], activation=activations[-1], name='hedge_qty_layer')
            )


    def call(self, feat_input, training=False):

        ten_dnn = self.lay_dnn_0(feat_input)
        for k in range(1, self.n_units - 1):
            ten_dnn = getattr(self, 'lay_dnn_' + str(k))(ten_dnn)

        ten_dnn = self.lay_hedge_qty(ten_dnn)

        return ten_dnn



class CashflowModel(tf.keras.Model):

    def __init__(self):

        super(CashflowModel, self).__init__()

        # hedge cashflows from trades
        self.dqty_inner = tf.keras.layers.Lambda(lambda x: x[:,1:] - x[:,:-1])        
        self.dqty0 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x[:,0,:], shape=(-1,1,1)))
        self.dqtyT = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(-x[:,-1,:], shape=(-1,1,1)))
        self.dqty = tf.keras.layers.Concatenate(axis=1)

        self.dqty_pos = tf.keras.layers.Lambda(lambda x: tf.keras.backend.minimum(-1.0*x, 0.0))
        self.dqty_neg = tf.keras.layers.Lambda(lambda x: tf.keras.backend.maximum(-1.0*x, 0.0))

        self.hedge_cashflow_buy = tf.keras.layers.Multiply()
        self.hedge_cashflow_sell = tf.keras.layers.Multiply()
        self.hedge_cashflow_trades = tf.keras.layers.Add(name='hedge_cashflow_trades')

        # hedge cashflows from payoffs
        self.qty_layer_initial = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(tf.zeros_like(x[:,0,:]),(-1,1,1)))
        self.qty_layer_extended = tf.keras.layers.Concatenate(axis=1)
        self.hedge_cashflow_payoff = tf.keras.layers.Multiply(name='hedge_cashflow_payoff')
        
        # total hedge cashflows
        self.hedge_cashflow = tf.keras.layers.Add(name='hedge_cashflow')

        # cumulated cashflows that define the loss
        self.cashflow = tf.keras.layers.Add(name='cashflow_total')

        # discounted cashflow defines the loss
        self.loss_timestep = tf.keras.layers.Multiply(name='cashflows_discounted')

        # aggregate loss output
        self.loss_value = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(tf.keras.backend.sum(x, axis=1), axis=-1, keepdims=True), name='loss')#(loss_timestep)


    def call(self, loss_inputs, training=False):
        
        # required order of loss_inputs = [hedge_qty, hedge_prc_bid, hedge_prc_ask, hedge_payoff, inst_payoff, disc_fac]
        hedge_qty = loss_inputs[0]
        hedge_prc_bid = loss_inputs[1]
        hedge_prc_ask = loss_inputs[2]
        hedge_payoff = loss_inputs[3]
        inst_payoff = loss_inputs[4]
        disc_fac = loss_inputs[5]

        # compute cumulated cashflows from hedge trades, hedge payoffs and instrument payoff
        dqty_inner = self.dqty_inner(hedge_qty)
        dqty0 = self.dqty0(hedge_qty)
        dqtyT = self.dqty0(hedge_qty)
        dqty = self.dqty([dqty0, dqty_inner, dqtyT])

        dqty_pos = self.dqty_pos(dqty)
        dqty_neg = self.dqty_neg(dqty)

        hedge_cf_buy = self.hedge_cashflow_buy([dqty_pos, hedge_prc_ask])
        hedge_cf_sell = self.hedge_cashflow_sell([dqty_neg, hedge_prc_bid])
        hedge_cf_trades = self.hedge_cashflow_trades([hedge_cf_buy, hedge_cf_sell])

        qty0 = self.qty_layer_initial(hedge_qty)
        qtyx = self.qty_layer_extended([qty0, hedge_qty])
        hedge_cf_payoff = self.hedge_cashflow_payoff([qtyx, hedge_payoff])

        hedge_cf = self.hedge_cashflow([hedge_cf_trades, hedge_cf_payoff])

        cashflow = self.cashflow([inst_payoff, hedge_cf])

        loss_timestep = self.loss_timestep([disc_fac, cashflow])
        loss_value = self.loss_value(loss_timestep)

        return loss_value




##===============================================================
## MAIN
##===============================================================

if __name__ == '__main__':

    # LOAD DATA
    #-------------------------
    feat_mkt, feat_prc, feat_aux = pickle.load(open(fpath, 'rb'))

    feat = {
        'TIME': feat_mkt['TIME'],
        'DF': np.cumprod(feat_mkt['EUR/DF_STEP'], axis=1),
        'DF_STEP': feat_mkt['EUR/DF_STEP'],
        'CF_STEP': feat_mkt['EUR/CF_STEP'],
        'SPOT': feat_mkt['DAI/SPOT'],
        'SPOT_BID': feat_mkt['DAI/SPOT'] * (1 - ba_spread * 0.5),
        'SPOT_ASK': feat_mkt['DAI/SPOT'] * (1 + ba_spread * 0.5),
        'BIDASK_SPREAD': feat_mkt['DAI/SPOT'] * ba_spread, 
        'ATMVOL1Y': feat_mkt['DAI/VOL/T360/X100'],
        'PAYOFF': feat_mkt['DAI/PAYOFF'],
        'CALL_TTM': feat_prc['DAI_S70.0_CALL_T90_X100/TTM'],
        'CALL_PRICE': feat_prc['DAI_S70.0_CALL_T90_X100/PRICE'],
        'CALL_DELTA': feat_prc['DAI_S70.0_CALL_T90_X100/DELTA/DAI'],
        'CALL_PAYOFF': feat_aux['DAI_S70.0_CALL_T90_X100/PAYOFF_CASHFLOW']
    }

    # PREPARE DATA
    #-------------------------
    n_time = len(feat['TIME'])
    n_feat = len(feat_def)
    n_sim = feat['SPOT'].shape[0]

    #--- Prepare Features Data
    f_hedge = np.zeros((n_sim, n_time, n_feat))
    
    for i, fid in enumerate(feat_def):
        f_hedge[:,:,i] = feat[fid]
    null_target = np.zeros((n_sim,1))

    f_loss_def = ['SPOT_BID', 'SPOT_ASK', 'PAYOFF', 'CALL_PAYOFF', 'DF']
    f_loss = [np.atleast_3d(feat[k]) for k in f_loss_def]
    
    ## THIS ONLY FOR DEBUG !!!
    #f_loss.insert(0, feat['CALL_DELTA'][:,:-1].reshape(-1,n_time-1,1))



    # BUILD AND TRAIN
    #-------------------------
    #---- delete later (begin)
    if False:
        hmdl = HedgeModel(n_feat, n_time, units, activations)
        hmdl.compile(optimizer='adam', loss='mean_squared_error')
        hmdl.fit(f_hedge[:,:-1,:], null_target)

        cmdl = CashflowModel()
        cmdl.compile()
        cmdl.call(f_loss)
    #---- delete later (end)

    model_param = {'units': units, 'activations': activations}
    hedger = Hedger(n_feat, n_time, model_param)

    optim = tf.keras.optimizers.Adam(lr=learning_rate)
    if loss_type == 'mean_squared_error':
        loss_fcn = 'mean_squared_error'
    else:
        loss_fcn = expected_shortfall_10pct
    hedger.compile(optimizer=optim, loss=loss_fcn)

    inputs = [f_hedge[:,:-1,:]] + f_loss
    hedger.fit(inputs, null_target, epochs=epochs, batch_size=batch_size, verbose=2)




    # EVALUATE AND TEST
    #-------------------------




    # PLOT AND ANALYSE
    #-------------------------


    print('Finished.')
