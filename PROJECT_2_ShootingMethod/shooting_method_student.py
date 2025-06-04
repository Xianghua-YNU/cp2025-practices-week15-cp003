def ode_system_shooting(t, y):
    """
    Define the ODE system for shooting method.
    
    Convert the second-order ODE u'' = -π(u+1)/4 into a first-order system:
    y1 = u, y2 = u'
    y1' = y2
    y2' = -π(y1+1)/4
    
    Args:
        t (float): Independent variable (time/position)
        y (array): State vector [y1, y2] where y1=u, y2=u'
    
    Returns:
        list: Derivatives [y1', y2']
    
    TODO: Implement the ODE system conversion
    Hint: Return [y[1], -np.pi*(y[0]+1)/4]
    """
    y = np.array(y, dtype=np.float64, ndmin=1)
    if len(y) < 2:
        y = np.append(y, 0.0)  # Append a zero if y has only one element
    
    return [y[1], -np.pi * (y[0] + 1) / 4]
