import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pandas as pd
import datetime as dt
import math
import pyvacon
import pyvacon.marketdata
import pyvacon.tools.enums as enums
import peer_group


n_spline_points = 20

class DataGenerator:
    def __init__(self,
                    disc_rate = 0.0,
                    borrow_rate = 0.0,
                    funding_rate = 0.0,
                    spot = 1.0,
                    vol = 0.3,
                    relative = False,
                n_splint_points = 20):
        self.relative = relative
        self.n_spline_points = n_splint_points
        self.refdate = dt.datetime(2020,4,1)
        self.spot = spot
        days_to_maturity = [1, 180, 365, 720, 3*365, 4*365, 10*365]
        dates = [self.refdate+dt.timedelta(days=d) for d in  days_to_maturity]
        # discount factors from constant rate
        self.dc = pyvacon.marketdata.DiscountCurve("TEST_DC", self.refdate, dates, [math.exp(-d/365.0*disc_rate) for d in days_to_maturity], 
                                              enums.DayCounter.ACT365_FIXED, enums.InterpolationType.HAGAN_DF, enums.ExtrapolationType.NONE)


        bc = pyvacon.marketdata.DiscountCurve('TEST_BC', self.refdate, dates, [math.exp(-d/365.0*borrow_rate) for d in days_to_maturity], enums.DayCounter.ACT365_FIXED, 
                                     enums.InterpolationType.HAGAN_DF, enums.ExtrapolationType.NONE)

        fc = pyvacon.marketdata.DiscountCurve('TEST_FC', self.refdate, dates, [math.exp(-d/365.0*funding_rate) for d in days_to_maturity], enums.DayCounter.ACT365_FIXED, 
                                     enums.InterpolationType.HAGAN_DF, enums.ExtrapolationType.NONE)

        #div table
        ex_dates = [dt.datetime(2018,3,29), dt.datetime(2019,3,29), dt.datetime(2020,3,29), dt.datetime(2021,3,29)]
        pay_dates = [dt.datetime(2018,4,1), dt.datetime(2019,4,1), dt.datetime(2020,4,1), dt.datetime(2021,4,1)]
        tax_factors = [1.0, 1.0, 1.0, 1.0]
        div_yield = [0, 0.000, 0.00, 0.00]
        div_cash = [0.0, 0.0, 0.0, 0.0]
        div_table=pyvacon.marketdata.DividendTable("Div_Table", self.refdate, ex_dates, div_yield, div_cash, tax_factors, pay_dates)
        #forward curve

        self.forward_curve = pyvacon.marketdata.EquityForwardCurve(self.refdate, spot, fc, bc, div_table)
        #vol surface
        flat_param = pyvacon.marketdata.VolatilityParametrizationFlat(vol)
        self.vol_surf = pyvacon.marketdata.VolatilitySurface('TEST_SURFACE', self.refdate, self.forward_curve, enums.DayCounter.ACT365_FIXED, flat_param)
        self.european_prdata = self._create_european_pricing_data()
        self.european_prdata2 = self._create_european_pricing_data()
        self.european_prdata3 = self._create_european_pricing_data()
        self.barrier_prdata = self._create_barrier_pricing_data()
    
    def _create_european_pricing_data(self):
        prdata = pyvacon.analytics.Black76PricingData()
        prdata.valDate = self.refdate
        prdata.dsc = self.dc
        prdata.param = pyvacon.analytics.PricingParameter()
        prdata.vol = self.vol_surf
        prdata.pricingRequest = pyvacon.analytics.PricingRequest()
        prdata.pricingRequest.setSpline(True)
        prdata.pricingRequest.setVega(True)
        prdata.pricingRequest.setTheta(True)
        return prdata
        #self.european_prdata = prdata
        
    def _create_barrier_pricing_data(self):
        prdata = pyvacon.analytics.LocalVolPdePricingData()
        prdata.valDate = self.refdate
        prdata.dsc = self.dc
        prdata.param = pyvacon.analytics.PdePricingParameter()
        prdata.vol = self.vol_surf
        prdata.pricingRequest = pyvacon.analytics.PricingRequest()
        prdata.pricingRequest.setSpline(True)
        prdata.pricingRequest.setVega(True)
        prdata.pricingRequest.setTheta(True)
        return prdata
        #self.barrier_prdata = prdata

    def _create_straddle(self, expiry, strike):
        
        prdata = pyvacon.analytics.ComboPricingData()
        
        self.european_prdata.spec = pyvacon.instruments.EuropeanVanillaSpecification('ID', 'DUMMY_ISSUER', 'COLLATERALIZED', 'EUR', 'DUMMY_UDL', 
                                                                                          'Call', expiry, strike*self.spot)
        self.european_prdata2.spec = pyvacon.instruments.EuropeanVanillaSpecification('ID', 'DUMMY_ISSUER', 'COLLATERALIZED', 'EUR', 'DUMMY_UDL', 
                                                                                          'Put', expiry, strike*self.spot)
        #print(expiry)
        ins = pyvacon.analytics.vectorBaseSpecification(2)
        ins[0] = self.european_prdata.spec
        ins[1] = self.european_prdata2.spec
        weights = pyvacon.analytics.vectorDouble([1.0,1.0])
        prdata.spec = pyvacon.analytics.ComboSpecification('', ins, weights)
        prdata.valDate = self.refdate        
        prdata.pricingData = pyvacon.analytics.vectorBasePricingData([self.european_prdata, self.european_prdata2])
        prdata.spec = pyvacon.analytics.ComboSpecification('', ins, weights)
        #print(prdata.spec.getExpiry().to_string())
        return prdata

    def _create_capped_call(self, expiry, strike, cap):
        
        prdata = pyvacon.analytics.ComboPricingData()
        
        self.european_prdata.spec = pyvacon.instruments.EuropeanVanillaSpecification('ID', 'DUMMY_ISSUER', 'COLLATERALIZED', 'EUR', 'DUMMY_UDL', 
                                                                                          'Call', expiry, strike*self.spot)
        self.european_prdata2.spec = pyvacon.instruments.EuropeanVanillaSpecification('ID', 'DUMMY_ISSUER', 'COLLATERALIZED', 'EUR', 'DUMMY_UDL', 
                                                                                          'Call', expiry, cap*self.spot)
        #print(expiry)
        ins = pyvacon.analytics.vectorBaseSpecification(2)
        ins[0] = self.european_prdata.spec
        ins[1] = self.european_prdata2.spec
        weights = pyvacon.analytics.vectorDouble([1.0,-1.0])
        prdata.spec = pyvacon.analytics.ComboSpecification('', ins, weights)
        prdata.valDate = self.refdate        
        prdata.pricingData = pyvacon.analytics.vectorBasePricingData([self.european_prdata, self.european_prdata2])
        prdata.spec = pyvacon.analytics.ComboSpecification('', ins, weights)
        #print(prdata.spec.getExpiry().to_string())
        return prdata

    def _create_capped_put(self, expiry, strike, cap):
        
        prdata = pyvacon.analytics.ComboPricingData()
        
        self.european_prdata.spec = pyvacon.instruments.EuropeanVanillaSpecification('ID', 'DUMMY_ISSUER', 'COLLATERALIZED', 'EUR', 'DUMMY_UDL', 
                                                                                          'Put', expiry, strike*self.spot)
        self.european_prdata2.spec = pyvacon.instruments.EuropeanVanillaSpecification('ID', 'DUMMY_ISSUER', 'COLLATERALIZED', 'EUR', 'DUMMY_UDL', 
                                                                                          'Put', expiry, cap*self.spot)
        #print(expiry)
        ins = pyvacon.analytics.vectorBaseSpecification(2)
        ins[0] = self.european_prdata.spec
        ins[1] = self.european_prdata2.spec
        weights = pyvacon.analytics.vectorDouble([1.0,-1.0])
        prdata.spec = pyvacon.analytics.ComboSpecification('', ins, weights)
        prdata.valDate = self.refdate        
        prdata.pricingData = pyvacon.analytics.vectorBasePricingData([self.european_prdata, self.european_prdata2])
        prdata.spec = pyvacon.analytics.ComboSpecification('', ins, weights)
        #print(prdata.spec.getExpiry().to_string())
        return prdata
    
    def _create_butterfly(self, expiry, strike, h):
        prdata = pyvacon.analytics.ComboPricingData()
        self.european_prdata.spec = pyvacon.instruments.EuropeanVanillaSpecification('ID', 'DUMMY_ISSUER', 'COLLATERALIZED', 'EUR', 'DUMMY_UDL', 
                                                                                          'Call', expiry, (strike-h)*self.spot)
        self.european_prdata2.spec = pyvacon.instruments.EuropeanVanillaSpecification('ID', 'DUMMY_ISSUER', 'COLLATERALIZED', 'EUR', 'DUMMY_UDL', 
                                                                                          'Put', expiry, strike*self.spot)
        self.european_prdata3.spec = pyvacon.instruments.EuropeanVanillaSpecification('ID', 'DUMMY_ISSUER', 'COLLATERALIZED', 'EUR', 'DUMMY_UDL', 
                                                                                          'Put', expiry, (strike+h)*self.spot)
        #print(expiry)
        ins = pyvacon.analytics.vectorBaseSpecification(3)
        ins[0] = self.european_prdata.spec
        ins[1] = self.european_prdata2.spec
        ins[2] = self.european_prdata3.spec
        weights = pyvacon.analytics.vectorDouble([1.0/h,-2.0/h, 1.0/h])
        prdata.spec = pyvacon.analytics.ComboSpecification('', ins, weights)
        prdata.valDate = self.refdate        
        prdata.pricingData = pyvacon.analytics.vectorBasePricingData([self.european_prdata, self.european_prdata2, self.european_prdata3])
        prdata.spec = pyvacon.analytics.ComboSpecification('', ins, weights)
        #print(prdata.spec.getExpiry().to_string())
        return prdata
        
    payoffs = ['C','CC', 'P','CP', 'UIC','DIC', 'DOC','UOC', 'UIP', 'DIP', 'DOP','UOP', 'STRADDLE']
    def create_input(self, payoff, ttm, barrierlevel, strike, barrier_start=0, barrier_end = None, rebate=0.0 ):
        expiry = self.refdate+dt.timedelta(days=ttm)
        if payoff == 'STRADDLE':
            prdata = self._create_straddle(expiry, strike)
        elif payoff == 'BUTTERFLY':
            prdata = self._create_butterfly(expiry, strike, barrierlevel)
        elif payoff == 'CC':
            prdata = self._create_capped_call(expiry, strike, barrierlevel)
        elif payoff == 'CP':
            prdata = self._create_capped_put(expiry, strike, barrierlevel)    
        elif payoff =='C' or payoff == 'P':
            p = 'Call'
            if payoff == 'P':
                p='Put'
            self.european_prdata.spec =  pyvacon.instruments.EuropeanVanillaSpecification('ID', 'DUMMY_ISSUER', 'COLLATERALIZED', 'EUR', 'DUMMY_UDL', 
                                                                                          p, expiry, strike*self.spot)
            prdata = self.european_prdata        
        else:
            if barrier_end is None:
                barrier_end = ttm
            if payoff[2] == 'C':
                p = DataGenerator._create_call_payoff(strike*self.spot)
            else:
                p = DataGenerator._create_put_payoff(strike*self.spot)
            p_rebate =  DataGenerator._create_rebate(rebate*self.spot)
            down = True
            if payoff[0] == 'U':
                down = False
            if payoff[1] =='O':
                barriers = self._create_barrier_schedule(barrier_start, barrier_end, 
                                                         barrierlevel, p_rebate, down, 
                                                         pyvacon.analytics.ptime())
                final_payoff = p
            else:
                barriers = self._create_barrier_schedule(barrier_start, barrier_end, 
                                                         barrierlevel, p, down, 
                                                         expiry)
                final_payoff = p_rebate
            self.barrier_prdata.spec =  pyvacon.analytics.BarrierSpecification('', 'DUMMY_ISSUER', "COLLATERALIZED", 'EUR', 'DUMMY_UDL', 
                                                  expiry, barriers, final_payoff)
            prdata = self.barrier_prdata
        return peer_group.compute_f_new(prdata, udl='DUMMY_UDL', relative=self.relative, n_points=self.n_spline_points)
        
    @staticmethod
    def _create_put_payoff(strike):
        xPoints = pyvacon.analytics.vectorDouble(3)
        pPoints = pyvacon.analytics.vectorDouble(3)
        xPoints[0] = 0
        xPoints[1] = strike
        xPoints[2] = strike +1
        pPoints[0] = strike
        pPoints[1] = 0
        pPoints[2] = 0
        result = pyvacon.analytics.PayoffStructure(xPoints, pPoints)
        return result
    @staticmethod
    def _create_call_payoff(strike):
        xPoints = pyvacon.analytics.vectorDouble(3)
        pPoints = pyvacon.analytics.vectorDouble(3)
        xPoints[0] = 0
        xPoints[1] = strike
        xPoints[2] = strike +1.0
        pPoints[0] = 0
        pPoints[1] = 0
        pPoints[2] = 1.0
        result = pyvacon.analytics.PayoffStructure(xPoints, pPoints)
        return result
    
    @staticmethod
    def _create_rebate(rebate):
        rebatePayoff = pyvacon.analytics.PayoffStructure(rebate)
        return rebatePayoff
    
    def _create_barrier_schedule(self, barrier_start, barrier_end, level, payoff, down, pays_at, calls = True):
        barrierPayoff = pyvacon.instruments.BarrierPayoff('', pays_at, payoff)
        barrier = pyvacon.instruments.BarrierDefinition(self.refdate+dt.timedelta(days=barrier_start), 
                                                            self.refdate+dt.timedelta(days=barrier_end),
                                                            barrierPayoff, self.spot*level, calls)
        result = pyvacon.analytics.BarrierSchedule()
        if down:
            result.addDownBarrier(barrier)
        else:
            result.addUpBarrier(barrier)
        return result


def _get_ins( payoff, ttm, barrierlevel, strike):
    return [payoff, ttm, barrierlevel, strike]
    
def _generate_strikes():
    anchor_strike = np.random.uniform(0.6, 1.4)
    pos_strike_dist = np.random.uniform(0.001, 0.5*anchor_strike)
    neg_strike_dist = np.random.uniform(pos_strike_dist+0.001, 2.0*pos_strike_dist)
    pos_strike = anchor_strike + np.random.choice([-1.0,1.0])*pos_strike_dist
    neg_strike = np.maximum(0.001, anchor_strike + np.random.choice([-1.0,1.0])*neg_strike_dist)
    return anchor_strike, pos_strike, neg_strike
    
def _generate_ttm():
    anchor_ttm = np.random.randint(20, 4*365)
    pos_ttm_dist = np.random.randint(5, int(anchor_ttm/2))
    neg_ttm_dist = np.random.randint(pos_ttm_dist+1, int(2*pos_ttm_dist))
    pos_ttm = anchor_ttm + np.random.choice([-1.0,1.0])*pos_ttm_dist
    neg_ttm = anchor_ttm + np.random.choice([-1.0,1.0])*neg_ttm_dist
    return anchor_ttm, pos_ttm, neg_ttm

def _random_uniform(lower, upper, v, w):
    """Creates a random uniform between lower and upper such that the random number has a larger distance to v then w
    
    Args:
        lower ([type]): [description]
        upper ([type]): [description]
        v ([type]): [description]
        w ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    dist=np.abs(v-w)
    if v-dist> lower and np.random.choice([True, False]):
        return np.random.uniform(lower, v-dist)
    return np.random.uniform(v+dist+0.000001, upper)

def _random_int(lower, upper, v, w):
    """Creates a random uniform int between lower and upper so that the result has a larger distance to v then w
    
    Args:
        lower ([type]): [description]
        upper ([type]): [description]
        v ([type]): [description]
        w ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    dist=np.abs(v-w)
    if v-dist > lower and np.random.choice([True, False]):
        return np.random.uniform(lower, v-dist)
    return np.random.uniform(v+dist+1, upper)

def European_European_Barrier(p, p_barrier):
    strikes = _generate_strikes()
    ttm = _generate_ttm()
    if p_barrier[0] == 'U':
        barrierlevel = np.random.uniform(1.01, 1.5)
    else:
        barrierlevel = np.random.uniform(0.6, 0.99)
    return _get_ins(p, ttm[0],-1,strikes[0]), _get_ins(p, ttm[1],-1,strikes[1]), _get_ins(p_barrier, ttm[2],-1,strikes[2])  
    
def UIC_UIC_UIC():
    strike = np.random.uniform(0.6,1.1)
    ttm = np.random.randint(20, 4*365)
    barrier = np.random.uniform(np.maximum(strike+0.01, 1.02), 1.35)
    barrier_1 = np.random.uniform(barrier+0.01, 1.4)
    barrier_2 = np.random.uniform(barrier_1+0.01, 1.45)
    return _get_ins('UIC', ttm,barrier,strike), _get_ins('UIC', ttm,barrier_1,strike), _get_ins('UIC', ttm,barrier_2,strike) 

def UOC_UOC_UOC():
    strike = np.random.uniform(0.6,1.1)
    ttm = np.random.randint(20, 4*365)
    barrier = np.random.uniform(np.maximum(strike+0.05, 1.05), 1.4)
    barrier_1 = np.random.uniform(barrier+0.01, 1.4)
    barrier_2 = np.random.uniform(barrier_1+0.01, 1.45)
    return _get_ins('UOC', ttm,barrier,strike), _get_ins('UOC', ttm,barrier_1,strike), _get_ins('UOC', ttm,barrier_2,strike) 

def UIC_UIC_UIC_2():
    strikes = _generate_strikes()
    ttm = np.random.randint(20, 4*365)
    barrier = np.random.uniform(np.maximum(np.max(strikes)+0.01, 1.02), 1.45)
    return _get_ins('UIC', ttm, barrier,strikes[0]), _get_ins('UIC', ttm,barrier,strikes[1]), _get_ins('UIC', ttm,barrier,strikes[2]) 

def UOC_UOC_UOC_2():
    strikes = _generate_strikes()
    ttm = np.random.randint(20, 4*365)
    barrier = np.random.uniform(np.maximum(np.max(strikes)+0.01, 1.02), 1.45)
    return _get_ins('UOC', ttm, barrier,strikes[0]), _get_ins('UOC', ttm,barrier,strikes[1]), _get_ins('UOC', ttm,barrier,strikes[2]) 


def UIC_UIC_UIC_3():
    strike = np.random.uniform(0.6,1.4)
    ttm = _generate_ttm()
    barrier = np.random.uniform(np.maximum(strike+0.001, 1.02), 1.45)
    return _get_ins('UIC', ttm[0], barrier,strike), _get_ins('UIC', ttm[1],barrier,strike), _get_ins('UIC', ttm[2],barrier,strike) 

def UOC_UOC_UOC_3():
    strike = np.random.uniform(0.6,1.4)
    ttm = _generate_ttm()
    barrier = np.random.uniform(np.maximum(strike+0.001, 1.02), 1.45)
    return _get_ins('UOC', ttm[0], barrier,strike), _get_ins('UOC', ttm[1],barrier,strike), _get_ins('UOC', ttm[2],barrier,strike) 


def UIC_C_C():
    strikes = _generate_strikes()
    ttm = np.random.randint(20, 4*365)
    barrier = np.random.uniform(np.maximum(np.max(strikes)+0.001, 1.02), 1.45)
    return _get_ins('UIC', ttm, barrier,strikes[0]), _get_ins('C', ttm,-1,strikes[1]), _get_ins('C', ttm,-1,strikes[2]) 

def C_UIC_UIC():
    strike = np.random.uniform(0.6,1.2)
    ttm = np.random.randint(20, 4*365)
    barrier_1 = np.random.uniform(np.maximum(strike+0.001, 1.02), 1.4)
    barrier_2 = np.random.uniform(barrier_1+0.05, 1.48)
    return _get_ins('C', ttm, -1,strike), _get_ins('UIC', ttm,barrier_1,strike), _get_ins('UIC', ttm,barrier_2,strike) 

def C_UOC_UOC():
    strike = np.random.uniform(0.6,1.2)
    ttm = np.random.randint(20, 4*365)
    barrier_2 = np.random.uniform(np.maximum(strike+0.001, 1.02), 1.4)
    barrier_1 = np.random.uniform(barrier_2+0.05, 1.48)
    return _get_ins('C', ttm, -1,strike), _get_ins('UOC', ttm,barrier_1,strike), _get_ins('UOC', ttm,barrier_2,strike) 


def DIP_DIP_DIP():
    strike = np.random.uniform(0.7,1.4)
    ttm = np.random.randint(20, 4*365)
    barrier = np.random.uniform(0.65, np.minimum(strike,0.98))
    barrier_1 = np.random.uniform(0.6, barrier)
    barrier_2 = np.random.uniform(0.55, barrier_1)
    return _get_ins('DIP', ttm,barrier,strike), _get_ins('DIP', ttm,barrier_1,strike), _get_ins('DIP', ttm, barrier_2,strike) 

def DIP_DIP_DIP_2():
    strikes = _generate_strikes()
    ttm = np.random.randint(20, 4*365)
    barrier = np.random.uniform(0.55, np.minimum(np.max(strikes),0.98))
    return _get_ins('DIP', ttm, barrier,strikes[0]), _get_ins('DIP', ttm,barrier,strikes[1]), _get_ins('DIP', ttm,barrier,strikes[2]) 

def DOP_DOP_P():
    strike = np.random.uniform(0.7,1.4)
    ttm =  np.random.randint(20, 4*365)
    barrier = np.random.uniform(0.65, np.minimum(strike,0.98))
    barrier_1 = np.random.uniform(0.60, barrier)
    return _get_ins('DOP', ttm,barrier,strike), _get_ins('DOP', ttm,barrier_1,strike), _get_ins('P', ttm, -1, strike) 

def DOP_DOP_DOP():
    strike = np.random.uniform(0.7,1.4)
    ttm =  np.random.randint(20, 4*365)
    barrier = np.random.uniform(0.65, np.minimum(strike,0.98))
    barrier_1 = np.random.uniform(0.60, barrier)
    return _get_ins('DOP', ttm,barrier,strike), _get_ins('DOP', ttm,barrier_1,strike), _get_ins('P', ttm, -1, strike) 


def European_P1_P1_P1():    
    p1 = np.random.choice(['C','P'])
    strikes = _generate_strikes()
    ttm = _generate_ttm()
    return _get_ins(p1, ttm[0],-1,strikes[0]), _get_ins(p1, ttm[1],-1,strikes[1]), _get_ins(p1, ttm[2],-1,strikes[2]) 
    
def CC_CC_CC():
    """Capped call condition
    """
    strike = np.random.uniform(0.7,1.3)
    ttm =  np.random.randint(20, 4*365)
    cap = np.random.uniform(strike+0.01, 1.3)
    cap_1 = np.random.uniform(cap+0.01, 1.35)
    cap_2 = np.random.uniform(cap_1+0.01, 1.4)
    return _get_ins('CC', ttm,cap,strike), _get_ins('CC', ttm,cap_1,strike), _get_ins('CC', ttm, cap_2, strike) 

def STRADDLE_STRADDLE_STRADDLE():
    p1 = 'STRADDLE'
    strikes = _generate_strikes()
    ttm = _generate_ttm()
    return _get_ins(p1, ttm[0],-1,strikes[0]), _get_ins(p1, ttm[1],-1,strikes[1]), _get_ins(p1, ttm[2],-1,strikes[2]) 
    
def BUTTERFLY_BUTTERFLY_BUTTERFLY():
    p1 = 'BUTTERLY'
    h = np.random.uniform(0.01,0.05)
    strikes = _generate_strikes()
    ttm = _generate_ttm()
    return _get_ins(p1, ttm[0],h,strikes[0]), _get_ins(p1, ttm[1],h,strikes[1]), _get_ins(p1, ttm[2],h,strikes[2]) 

def BUTTERFLY_BUTTERFLY_BUTTERFLY_2():
    p1 = 'BUTTERLY'
    strike = np.random.uniform(0.6,1.4)
    ttm = np.random.randint(20, 4*365)
    h = np.random.uniform(0.01,0.05)
    h1 = h+np.random.uniform(0.005,0.02)
    h2 = h1 + np.random.uniform(0.0025, 0.015)
    return _get_ins(p1, ttm,h,strike), _get_ins(p1, ttm,h1,strike), _get_ins(p1, ttm,h2,strike) 
    
def STRADDLE_P1_P2():
    strikes = _generate_strikes()
    ttm = _generate_ttm()
    anchor = _get_ins('STRADDLE', ttm[0],-1,strikes[0])
    positive =  _get_ins(np.random.choice(['C','P']), ttm[1],-1,strikes[1])
    negative = _get_ins(np.random.choice(['C','P']), ttm[2],-1,strikes[2])
    return anchor, positive, negative

def European_P1_P1_P2():
    p1 = np.random.choice(['C','P'])
    if p1 == 'C':
        p2 = 'P'
    else:
        p2 = 'C' 
    anchor = _get_ins(p1, np.random.randint(20, 4*365),-1,np.random.uniform(0.6, 1.4))
    positive =  _get_ins(p1, np.random.randint(20, 4*365),-1,np.random.uniform(0.6, 1.4))
    negative = _get_ins(p2, np.random.randint(20, 4*365),-1,np.random.uniform(0.6, 1.4))
    return anchor, positive, negative

def European_P1_P2_P2():
    p1 = np.random.choice(['C','P'])
    if p1 == 'C':
        p2 = 'P'
    else:
        p2 = 'C' 
    strikes = _generate_strikes()
    ttm = _generate_ttm()
    return _get_ins(p1, ttm[0],-1,strikes[0]), _get_ins(p2, ttm[1],-1,strikes[1]), _get_ins(p2, ttm[2],-1,strikes[2]) 
     

def generate_samples(file_prefix, samples, vol = 0.3):
    result = []
    for s in samples:
        for i in range(s[1]):
            anchor, positive, negative = s[0]()
            result.append((anchor+positive)+negative)        
    info_columns = []
    for prefix in ['anchor_', 'positive_', 'negative_']:
        for c in ['payoff', 'ttm', 'barrierlevel', 'strike']:
            info_columns.append(prefix+c)
    s = pd.DataFrame(result, columns=info_columns)
    s.to_csv(file_prefix+'_info.csv', index=False)
    
    values = s.values
    anchor = np.empty((values.shape[0],n_spline_points+3))
    positive = np.empty(anchor.shape)
    negative = np.empty(anchor.shape)
    data_gen = DataGenerator(n_splint_points=n_spline_points, vol=vol)
    
    distance = np.empty((anchor.shape[0],4,))
    for i in range(values.shape[0]):
        anchor[i,:] = data_gen.create_input(*values[i,0:4])
        positive[i,:] = data_gen.create_input(*values[i,4:8])
        negative[i,:] = data_gen.create_input(*values[i,8:12])
        
        distance[i,0] = peer_group.compute_peer_distance(anchor[i,:-3], positive[i,:-3] )
        distance[i,1] = peer_group.compute_peer_distance(anchor[i,:-3], negative[i,:-3] )
        if distance[i,0] <distance[i,1]:
            distance[i,2] = 1.0
            distance[i,3] = distance[i,1] - distance[i,0]
        else:
            distance[i,2] = 0.0
            distance[i,3] = 0.0
    value_columns = ['spot_'+str(i) for i in range(n_spline_points)]
    value_columns.extend(['vega','theta', 'ttm'])
    neg_v = pd.DataFrame(negative,columns=value_columns)
    anchor_v = pd.DataFrame(anchor,columns=value_columns)
    positive_v = pd.DataFrame(positive,columns=value_columns)
    positive_v.to_csv(file_prefix+'_positive.csv', index=False)
    anchor_v.to_csv(file_prefix+'_anchor.csv', index=False)
    neg_v.to_csv(file_prefix+'_negative.csv', index=False)
    distance_v = pd.DataFrame(distance,columns=['d_pos', 'd_neg', 'd_pos_less_d_neg', 'alpha'])
    distance_v.to_csv(file_prefix+'_distance.csv', index=False)
    

