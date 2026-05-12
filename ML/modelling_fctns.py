import numpy as np

def phase_dependent_response(driver_values, t_dev, responses, thresholds):
    #Thresholds are the thresholds in development time where the different growth phases change
    #Responses are the response functions, index starting at 'before the first threshold'
    #driver values are the inputs to the response function
    #t_dev is the (cts) development time
    phase = np.digitize(t_dev, thresholds)
    response = np.zeros(driver_values.shape)
    for phase_index in range(len(responses)):
        response += (phase == phase_index)*responses[phase_index](driver_values) #First brackets indicates if we are at the right phase, second takes the response function for each phase
    return response

def Wang_Engel_Temp_response(T, T_min, T_opt, T_max, beta = 1):
    alpha = np.log(2)/np.log( (T_max - T_min)/(T_opt - T_min) )
    f_T = ( ( (2*(np.sign(T - T_min)*(T - T_min))**alpha)*((T_opt - T_min)**alpha) - ((np.sign(T - T_min)*(T - T_min))**(2*alpha)) ) / ((T_opt - T_min)**(2*alpha)) )**beta
    f_T = np.nan_to_num(f_T)
    return f_T*(T >= T_min)*(T<= T_max)

def Trapezoid_Temp_response(T, T_min, T_opt1, T_opt2, T_max):
    pre_opt = (T>=T_min)*(T<=T_opt1)
    opt = (T>=T_opt1)*(T<=T_opt2)
    post_opt = (T>=T_opt2)*(T<=T_max)
    return pre_opt*(T - T_min)/(T_opt1 - T_min) + opt + post_opt*(T_max - T)/(T_max - T_opt2) 

def Trapezoid_Temp_derivs(T, T_min, T_opt1, T_opt2, T_max):
    pre_opt = (T>=T_min)*(T<=T_opt1)*np.array([(T - T_min)/(T_opt1 - T_min),
                                               (T - T_opt1)/((T_opt1 - T_min)**2),
                                               (T_min - T)/((T_opt1 - T_min)**2),
                                               np.zeros(T.shape),
                                               np.zeros(T.shape)])
    opt = (T>=T_opt1)*(T<=T_opt2)*np.array([np.ones(T.shape),
                                            np.zeros(T.shape),
                                            np.zeros(T.shape),
                                            np.zeros(T.shape),
                                            np.zeros(T.shape)])
    post_opt = (T>=T_opt2)*(T<=T_max)*np.array([(T - T_min)/(T_opt1 - T_min),
                                               (T - T_opt1)/((T_opt1 - T_min)**2),
                                               (T_min - T)/((T_opt1 - T_min)**2),
                                               np.zeros(T.shape),
                                               np.zeros(T.shape)])
    if pre_opt:
        d_dScale = (T - T_min)/(T_opt1 - T_min)
        d_dTmin = (T - T_opt1)/((T_opt1 - T_min)**2)
        d_dTopt1 = (T_min - T)/((T_opt1 - T_min)**2)
        d_dTopt2 = np.zeros(T.shape)
        d_dTmax = np.zeros(T.shape)
    elif opt:
        return []
    elif post_opt:
        d_dScale = (T_max - T)/(T_max - T_opt2) 
        d_dTmin = np.zeros(T.shape)
        d_dTopt1 = np.zeros(T.shape)
        d_dTopt2 = (T_max - T)/((T_max - T_opt2)**2)
        d_dTmax = (T - T_opt2)/((T_max - T_opt2)**2)
    return pre_opt + opt + post_opt