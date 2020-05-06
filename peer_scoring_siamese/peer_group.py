import numpy as np
import pyvacon

from distance import *


def set_spline_params(pricing_data, spot_min, spot_max, n_points):
    p_data = pyvacon.analytics.ComboPricingData.create_from(pricing_data)
    if p_data is not None:
        for x in p_data.pricingData:
            set_spline_params(x, spot_min, spot_max, n_points)
        return p_data
    p_data = pyvacon.analytics.LocalVolPdePricingData.create_from(pricing_data)
    if p_data is None:
        p_data = pyvacon.analytics.Black76PricingData.create_from(pricing_data)
    elif p_data is None:
        raise Exception('Unknown type')
    p_data.param.spotMinSpline = spot_min
    p_data.param.spotMaxSpline = spot_max
    p_data.param.nSplinePoints = n_points
    return p_data

def compute_f(pricing_data, min_spot = 0.55, max_spot = 1.45, n_points=80, udl = '', relative=True):
    pr = pyvacon.analytics.PricingRequest()
    pr.setSpline(True)
    pr.setVega(True)
    pr.setTheta(True)
    new_x = np.arange(0.6, 1.4, (1.4-0.6)/n_points)
    set_spline_params(pricing_data, min_spot, max_spot, n_points)
    pricing_data.pricingRequest = pr
    result = pyvacon.analytics.price(pricing_data)
    x = pyvacon.analytics.vectorDouble()
    y = pyvacon.analytics.vectorDouble()
    result.getSplineX(x)
    result.getSplineY(y)
    spot = result.getSpot(udl)
    price = result.getCleanPrice()
    for j in range(len(x)):
        x[j] /= spot
        if relative:
            y[j] /= np.maximum(price, 0.0001)
    vega = result.getVega1D()
    theta = result.getTheta()
    if relative:
        vega /= np.maximum(price, 0.0001)
        theta /= np.maximum(price, 0.0001) 
    return np.interp(new_x, x, y), vega, theta

def compute_f_new(pricing_data, min_spot = 0.55, max_spot = 1.45, n_points=80, udl = '', relative=True):
    pr = pyvacon.analytics.PricingRequest()
    pr.setSpline(True)
    pr.setVega(True)
    pr.setTheta(True)
    new_x = np.arange(0.6, 1.4, (1.4-0.6)/n_points)
    pricing_data = set_spline_params(pricing_data, min_spot, max_spot, n_points)
    pricing_data.pricingRequest = pr
    result = pyvacon.analytics.price(pricing_data)
    x = pyvacon.analytics.vectorDouble()
    y = pyvacon.analytics.vectorDouble()
    result.getSplineX(x)
    result.getSplineY(y)
    spot = result.getSpot(udl)
    price = result.getCleanPrice()
    for j in range(len(x)):
        x[j] /= spot
        if relative:
            y[j] /= np.maximum(price, 0.0001)
    vega = result.getVega1D()
    theta = result.getTheta()
    dc = pyvacon.analytics.DayCounter('ACT365FIXED')
    ttm = dc.yf(pricing_data.valDate, pricing_data.spec.getExpiry())
    if relative:
        vega /= np.maximum(price, 0.0001)
        theta /= np.maximum(price, 0.0001) 
    return np.concatenate([np.interp(new_x, x, y), [vega, theta, ttm]])

def compute_similarity_matrix(f):
    result = np.ones((f.shape[0], f.shape[0],))
    dx_f = f[:, 1:]-f[:, :-1]
    d = np.sqrt(np.sum(dx_f*dx_f, axis=1))

    for i in range(f.shape[0]):
        for j in range(i, f.shape[0]):
            if d[i] < 0.00001 or d[j] < 0.00001:
                if d[j] > 0.0001 or d[i] > 0.0001:
                    # no similarity: Constant and non-constant
                    result[i, j] = 0.0
                else:
                    # very similar since both seem to be constants
                    result[i, j] = 0.98
            else:
                result[i, j] = np.minimum(
                    np.dot(dx_f[i, :], dx_f[j, :])/(d[i]*d[j]), 1.0)
            result[j, i] = result[i, j]
    return result

def compute_distance_matrix(f, vega=None, theta=None):
    result = compute_similarity_matrix(f)
    #dx_f = f[:, 1:]-f[:, :-1]
    for i in range(f.shape[0]):
        for j in range(i, f.shape[0]):        
            result[i,j] = ( 1.0-result[i,j]+ 0.001)*(1.0 + 10.0*np.abs(theta[i]-theta[j]))*(1.0+ 10.0*np.abs(vega[i]-vega[j]))*np.linalg.norm(f[i,:]-f[j,:])
            result[j, i] = result[i, j]
    return result

def compute_distance_matrix_1(f, vega=None, theta=None, vega_scale = 1.0, theta_scale=1-0):
    result = np.empty((f.shape[0], f.shape[0],))
    for i in range(f.shape[0]):
        for j in range(i, f.shape[0]):        
            #result[i,j] = ( 1.0-result[i,j]+ 0.001)*(1.0 + 10.0*np.abs(theta[i]-theta[j]))*(1.0+ 10.0*np.abs(vega[i]-vega[j]))*np.linalg.norm(f[i,:]-f[j,:])
            result[i,j] = np.linalg.norm(f[i,:]-f[j,:])/f.shape[1] + theta_scale*np.abs(theta[i]-theta[j]) + vega_scale*np.abs(vega[i]-vega[j])
            result[j, i] = result[i, j]
    return result
    
    