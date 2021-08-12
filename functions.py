


def odePressure(t, P, q, a, b, P0):
    ''' Return the derivative dP/dt at a time, t for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        P : float
            Dependent varaible.
        q : float
            Source/sink rate.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        P0 : float
            Initial Pressure value.

        Returns:
        --------
        dPdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        >>> 

    '''

    dPdt = -a*q - b*(P - P0)
    return dPdt