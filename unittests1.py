# 263 Project: Group 5 
# Unit Tests 

import numpy as np
from os import sep
from matplotlib import pyplot as plt
from numpy.core.defchararray import array
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from benchmark import*
from main import*

Test_Tolerance= 1.e-10

Q = Qterms()

def Test_odePressure():
    """
    Testing the odePressure function with known results to ensure it is working properly. 
    """

    # Test odePressure with all inputs as '0'.
    test_odePressure1 = odePressure(0, 0, Q.giveQ, 0, 0, 0)
    assert((test_odePressure1 - 0.0) < Test_Tolerance)

    # Test odePressure with all inputs as infinity.
    try:
        test_odePressure2 = odePressure(np.inf, np.inf, Q.giveQ, np.inf, np.inf, np.inf)
    except UnboundLocalError:
        pass
    
          
    # Test odePressure with arbitrary values.
    test_odePressure3 = odePressure(5, 8, Q.giveQ, 9, 7, 4)
    assert((test_odePressure3 + 28.0) < Test_Tolerance)

    # Test odePressure with string input.
    try:
        test_odePressure4 = odePressure("computational_modelling", 0, Q.giveQ, 0, 0, 0)
    except TypeError:
        pass

    # Test odePressure without getting Q value.
    try:
        test_odePressure5 = odePressure(0, 0, 0, 0, 0, 0)
    except TypeError:
        pass

    # Test odePressure with array input.
    try:
        test_odePressure6 = odePressure([0,0,0,0], 0, Q.giveQ, 0, 0, 0)
    except ValueError:
        pass

    # Test odePressure with negative arbitrary values.
    test_Pressure7 = odePressure(-50, -60, Q.giveQ, -10, -20, -30)
    assert((test_Pressure7 + 600.0) < Test_Tolerance)

def Test_odeTemp():
    """
    Testing the odeTemperature function with known results to ensure it is working properly. 
    """

    # Test odeTemp with all inputs as '0'.
    try:
        test_odeTemp1 = odeTemp(0, 0, 0, Q.giveQs, 0, 0, 0, 0, 0, 0)
    except ZeroDivisionError:
        pass

    # Test odeTemp with all inputs as infinity.
    try:
        test_odeTemp2 = odeTemp(np.inf, np.inf, np.inf, Q.giveQs, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
    except UnboundLocalError:
        pass

    # Test odeTemp with arbitrary values.
    test_odeTemp3 = odeTemp(50, 60, 70, Q.giveQs, 10, 20, 30, 40, 50, 60)
    assert((test_odeTemp3 - 2099.0767688954033) < Test_Tolerance)

    # Test odeTemp with string input.
    try:
        test_odeTemp4 = odeTemp("computational modelling", 0, 0, Q.giveQs, 0, 0, 0, 0, 0, 0)
    except TypeError:
        pass

    # Test odeTemp with array input.
    try:
        test_odeTemp5 = odeTemp([0,0,0,0], 0, 0, Q.giveQs, 0, 0, 0, 0, 0, 0)
    except ValueError:
        pass

    # Test odeTemp without getting Qsteam value.
    try:
        test_odeTemp6 = odeTemp(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    except TypeError:
        pass

    # Test odeTemp with negative arbitrary values.
    test_odeTemp7 = odeTemp(-50, -60, -70, Q.giveQ, -10, -20, -30, -40, -50, -60)
    assert((test_odeTemp7 - 0.0) < Test_Tolerance)

def Test_Tprime():
    """
    Testing the Tprime function with known results to ensure it is working properly. 
    """
    # Test Tprime with all inputs as '0'.
    test_Tprime1 = Tprime(0, 0, 0, 0, 0)
    assert((test_Tprime1 - 0.0) < Test_Tolerance)

    # Test Tprime where P > P0.
    test_Tprime2 = Tprime(1, 4, 5, 2, 3)
    assert((test_Tprime2 - 5) < Test_Tolerance)

    # Test Tprime where P < P0.
    test_Tprime3 = Tprime(1, 2, 5, 4, 3)
    assert((test_Tprime3 - 3) < Test_Tolerance)

    # Test Tprime where P = P0.
    test_Tprime4 = Tprime(1, 6, 5, 6, 3)
    assert((test_Tprime4 - 3) < Test_Tolerance)

    # Test Tprime with string inputs.
    try:
        test_Tprime5 = Tprime(0, "computational modelling", 0, 0, 0)
    except TypeError:
        pass

    # Test Tprime with array inputs.
    try:
        test_Tprime6 = Tprime(0, [0, 0, 0, 0], 0, 0, 0)
    except TypeError:
        pass

def Test_solvePressure():
    """
    Testing the solvePressure function with known results to ensure it is working properly. 
    """

    # Test solvePressure with all inputs as '0'. 
    pars = [0,0,0]
    test_solvePressure1 = solvePressure([0,0,0], 0, 0, Q.giveQ, pars)
    assert(test_solvePressure1[0] == [0, 0, 0])
    assert(any(test_solvePressure1[1]) == any([0, 0, 0]))

    pars1 = [1,2,3]

    # Test solvePressure with inputs as inf.
    try: 
        test_solvePressure2 = solvePressure([np.inf, np.inf, np.inf], np.inf, np.inf, Q.giveQ, pars1)
    except UnboundLocalError:
        pass
    
    # Test solvePressure with inputs as '1'.
    pars2 = [1,1,1]
    test_solvePressure3 = solvePressure([1,1,1], 1, 1, Q.giveQ, pars2)
    assert(test_solvePressure3[0] == [1, 1, 1])
    assert(any(test_solvePressure3[1]) == any([1, 1, 1]))

    pars3 = [1,2,3]

    # Test solvePressure with string inputs. 
    try:
        test_solvePressure4 = solvePressure([1,1,1], "computational modelling", 1, Q.giveQ, pars3)
    except TypeError:
        pass

    # Test solvePressure with incorrect array inputs. 
    try:
        test_solvePressure5 = solvePressure([1,1,1], [1,1,1], 1, Q.giveQ, pars3)
    except TypeError:
        pass

    # Test solvePressure with negative numbers.
    pars4 = [-1,-2,-3]
    test_solvePressure6 = solvePressure([-1,-1,-1], -1, 1, Q.giveQ, pars4)
    assert(test_solvePressure6[0] == [-1, -1, -1])
    assert(any(test_solvePressure6[1]) == any([-1, -1, -5]))

    # Test solvePressure without getting Q value. 
    pars5 = [1, 2, 3]
    try:
        test_solvePressure7 = solvePressure([1,1,1], 1, 1, 1, pars5)
    except TypeError:
        pass

    # We don't have a mathematically worked out test, do we need one? 

def Test_solveTemperature():
    """
    Testing the solveTemperature function with known results to ensure it is working properly. 
    """
    # Test solveTemperature with all inputs as '0'. 
    try:
        test_solveTemperature1 = solveTemperature([0,0,0], 0, [0, 0, 0,], Q.giveQs, 0, 0, 0, 0, 0, 0)
    except ZeroDivisionError:
        pass

    # Test solveTemperature with inputs as inf.
    try: 
        test_solveTemperature2 = solveTemperature([np.inf, np.inf, np.inf], np.inf, [np.inf, np.inf, np.inf], Q.giveQs, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
    except UnboundLocalError:
        pass
    
    # Test solveTemperature with inputs as '1'.
    test_solveTemperature3 = solveTemperature([1,1,1], 1, [1, 1, 1], Q.giveQ, 1, 1, 1, 1, 1, 1)
    assert(test_solveTemperature3[0] == [1, 1, 1])
    assert(any(test_solveTemperature3[1]) == any([1, 1, 1]))

    # Test solveTemperature with string inputs. 
    try:
        test_solveTemperature4 = solveTemperature([1,1,1], "computational modelling", [1, 1, 1], Q.giveQ, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    # Test solveTemperature with incorrect array inputs. 
    try:
        test_solveTemperature5 = solveTemperature([1,1,1], [1, 1, 1], [1, 1, 1], Q.giveQ, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    # Test solveTemperature with negative numbers.
    test_solveTemperature6 = solveTemperature([-1,-1,-1], -1, [-1, -1, -1], Q.giveQ, -1, -1, -1, -1, -1, -1)
    assert(test_solveTemperature6[0] == [-1, -1, -1])
    assert(any(test_solveTemperature6[1]) == any([-1, -1, -1]))

    # Test solveTemperature with arbitrary numbers. 
    test_solveTemperature7 = solveTemperature([5,7,2], 20, [10,1,8], Q.giveQs, 1, 2, 3, 4, 5, 6)
    assert(test_solveTemperature7[0] == [5, 7, 2])
    assert(any(test_solveTemperature7[1]) == any([6, 6, 6]))

    # Test solveTemperature without getting Qsteam value. 
    try:
        test_solveTemperature8 = solveTemperature([1,1,1], 1, [1, 1, 1], 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass
    
def main():
    Test_odeTemp()
    Test_odePressure()
    Test_Tprime()
    Test_solvePressure()
    Test_solveTemperature()

if __name__=="__main__":
    main()

