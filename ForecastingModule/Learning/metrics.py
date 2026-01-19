import torch
import copy

# Metrics

'''
Metrics evaluate SIC
'''
'''
Tensor Shape: t x h x w
'''
def MSE_Metric(pred,gt, channel_wise_res=False):
    '''
    The lower the better.

    For channel_wise_res == True:
        pred,gt : [C, H=448, W=304]

    For channel_wise_res == False:
        pred,gt : [1, H=448, W=304]

    '''
    y_hat = copy.deepcopy(pred)
    y = copy.deepcopy(gt)

    if channel_wise_res:
        mse = []
        for y_hat_t, y_t in zip(y_hat,y):
            mse.append(torch.nn.MSELoss(y_hat_t,y_t))
        return  mse, torch.mean(mse)
    else:
        return torch.nn.MSELoss(y_hat_t,y_t)



def RMSE_Metric(pred,gt, channel_wise_res=False):
    '''
    The lower the better.

    For channel_wise_res == True:
        pred,gt : [C, H=448, W=304]

    For channel_wise_res == False:
        pred,gt : [1, H=448, W=304]

    '''
    y_hat = copy.deepcopy(pred)
    y = copy.deepcopy(gt)
    if channel_wise_res:
        rmse = []
        for y_hat_t, y_t in zip(y_hat,y):
            mse = torch.nn.MSELoss(y_hat_t,y_t)
            rmse.append(torch.rsqrt(mse))
        return  rmse, torch.mean(rmse)
    else:
        return torch.nn.MSELoss(y_hat_t,y_t)



def MAE_Metric(pred,gt, channel_wise_res=False):
    '''
    The lower the better.

    For channel_wise_res == True:
        pred,gt : [C, H=448, W=304]

    For channel_wise_res == False:
        pred,gt : [1, H=448, W=304]

    '''
    y_hat = copy.deepcopy(pred)
    y = copy.deepcopy(gt)
    
    if channel_wise_res:
        mae = []
        for y_hat_t, y_t in zip(y_hat,y):
            mae.append(torch.nn.L1Loss(y_hat_t,y_t))
        return  mae, torch.mean(mae)
    else:
        return torch.nn.L1Loss(y_hat_t,y_t)


def R_Squared_Metric(pred,gt, channel_wise_res=False):
    '''
    Coefficient of Determination

    R_Squared = 1 - RSS/TSS

    RSS: The Sum of Squares of Residuals.
    TSS: Total Sum of Squares.

    For channel_wise_res == True:
        y_hat,y : [C, H=448, W=304]

    For channel_wise_res == False:
        y_hat,y : [1, H=448, W=304]

    '''
    y_hat = copy.deepcopy(pred)
    y = copy.deepcopy(gt)

    if channel_wise_res:
        r_squared = []
        for y_hat_t, y_t in zip(y_hat,y):
            rss = torch.sum(torch.square(y_hat_t - y_t))
            tss = torch.sum(torch.square(y_t))
            r_squared.append(1- rss/tss)
        return  r_squared, torch.mean(r_squared)
    
    else:
        r_squared = 0.
        rss = torch.sum(torch.square(y_hat - y))
        tss = torch.sum(torch.square(y))
        r_squared = 1 - rss/tss
    
        return r_squared



def NSE_Metric(pred,gt):
    '''
    Nash-Sutcliffe Efficiency

    NSE = 1 - SUM( (actual SIC - predicted SIC)^2 ) / SUM ( (actual SIC - mean(actual SIC))^2 )

    '''
    y_hat = copy.deepcopy(pred)
    y = copy.deepcopy(gt)
    
    nse = 0.

    dividend = torch.sum(torch.square(y - y_hat))
    divisor  = torch.sum(torch.square(y - torch.mean(y)))

    nse = 1 - dividend/divisor
    return nse

'''
Metrics evaluate Sea Ice Extent
'''

def SIE_Metric(pred,gt, data_cap=1, area_scaler=None):
    '''
    SIE Calculation for a single SIC map
    without Land, Ocean Masks

    pred:  [Channel=1, H=448, W=304]
    gt    :  [Channel=1, H=448, W=304]

    max sic diff: 448*304
    data_cap: The original SIC value (0-1) could be multiplied by a certain number, say 2.5.

    
    Proposed SIE evaluation Metric:
        Aera difference the lower the better.
        IoU the higher the better.

    '''
    area_diff = 0
    iou = 0

    y_hat = copy.deepcopy(pred)
    y = copy.deepcopy(gt)

    sie_edge_value = 0.15 * data_cap

    # Area
    y_hat_area_above_15_percent = y_hat[y_hat > sie_edge_value]
    y_area_15_above_percent = y[y > sie_edge_value]
    area_diff = torch.abs(torch.tensor(y_hat_area_above_15_percent.size()[0] - y_area_15_above_percent.size()[0]))

    # Intersection of Union (The Binary Class Case)
    identifier = 10 * data_cap
    intersection_id = 2 * identifier - 1
    union_id = identifier - 1
    y_hat_area_above_15_percent = y_hat
    y_hat_area_above_15_percent[y_hat > sie_edge_value] = identifier
    y_area_15_above_percent = y
    y_area_15_above_percent[y > sie_edge_value] = identifier

    intersection = union = y_area_15_above_percent + y_hat_area_above_15_percent
    intersection = intersection[intersection > intersection_id].size()[0] # Avoid imprecise equal in Float Type

    union = union[union > union_id].size()[0]

    iou = torch.tensor(intersection/union) # IoU ranges from 0 to 1

    if area_scaler is not None:
        area_diff = area_diff * area_scaler # area_scaler scales area diff to match training loss.

    return area_diff, iou


def IIEE_Metric(pred,gt, data_cap=1):
    '''
    Integrated Ice Edge Error score
    
    IIEE = Overestimated + Underestimated Sea Ice Area
    
    '''
    y_hat = copy.deepcopy(pred)
    y = copy.deepcopy(gt)

    iiee = 0.
    sie_edge_value = 0.15 * data_cap
    forecasted_ice_aera = y_hat
    true_ice_area = y

    id = 20 * data_cap
    y_hat_ice_area_filter = y_hat > sie_edge_value
    y_ice_area_filter = y > sie_edge_value

    forecasted_ice_aera[:,:] = 0.
    true_ice_area[:,:] = 0.

    forecasted_ice_aera[y_hat_ice_area_filter] = id
    true_ice_area[y_ice_area_filter] = id

    difference = forecasted_ice_aera - true_ice_area
    
    overestimated_SIE = difference[difference > 0].size()[0]
    underestimated_SIE = difference[difference < 0].size()[0]

    iiee = overestimated_SIE + underestimated_SIE

    return iiee
