from numpy import zeros, percentile, array, histogram, hstack, round, linspace
from scipy import stats
from scipy.stats import chi2, ks_2samp, kstest
from matplotlib import pyplot as plt
from pickle import load

if __name__ == '__main__':

    curvedDis = load(open('curved_dis.p', 'rb'))
    curvedDon = load(open('curved_don.p', 'rb'))
    smoothDis = load(open('smooth_dis.p', 'rb'))
    smoothDon = load(open('smooth_don.p', 'rb'))
    texturedDis = load(open('textured_dis.p', 'rb'))
    texturedDon = load(open('textured_don.p', 'rb'))

    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    pCurvedDis = percentile(curvedDis, percentiles)
    pCurvedDon = percentile(curvedDon, percentiles)
    pSmoothDis = percentile(smoothDis, percentiles)
    pSmoothDon = percentile(smoothDon, percentiles)
    pTexturedDis = percentile(texturedDis, percentiles)
    pTexturedDon = percentile(texturedDon, percentiles)

    plt.figure()
    plt.plot(percentiles, pSmoothDis, color='r', label='smooth')
    plt.plot(percentiles, pTexturedDis, color='g', label='textured')
    plt.plot(percentiles, pCurvedDis, color='b', label='curved')
    plt.xlabel('Percentile [%]')
    plt.ylabel('Distance from Plane [m]')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.yticks([0.001, 0.01, 0.1, 0.5, 1], [0.001, 0.01, 0.1, 0.5, 1.0])

    plt.figure()
    plt.plot(percentiles, pSmoothDon, color='r', label='smooth')
    plt.plot(percentiles, pTexturedDon, color='g', label='textured')
    plt.plot(percentiles, pCurvedDon, color='b', label='curved')
    plt.xlabel('Percentile [%]')
    plt.ylabel('Norm Difference of Normals [-]')
    plt.legend(loc='best')

    plt.show()

    tests = []
    tests.append(ks_2samp(smoothDis, texturedDis))
    tests.append(ks_2samp(smoothDis, curvedDis))
    tests.append(ks_2samp(texturedDis, curvedDis))
    tests.append(ks_2samp(smoothDon, texturedDon))
    tests.append(ks_2samp(smoothDon, curvedDon))
    tests.append(ks_2samp(texturedDon, curvedDon))

    tests.append(kstest(smoothDis, 'chi2', args=(1, smoothDis.mean(), smoothDis.std()), N=1000))
    tests.append(kstest(smoothDon, 'chi2', args=(1, smoothDon.mean(), smoothDon.std()), N=1000))

    tests.append(kstest(texturedDis, 'chi2', args=(1, texturedDis.mean(), texturedDis.std()), N=1000))
    tests.append(kstest(texturedDon, 'chi2', args=(1, texturedDon.mean(), texturedDon.std()), N=1000))

    tests.append(kstest(curvedDis, 'chi2', args=(1, curvedDis.mean(), curvedDis.std()), N=1000))
    tests.append(kstest(curvedDon, 'chi2', args=(1, curvedDon.mean(), curvedDon.std()), N=1000))

    print(array(tests))

    x = linspace(chi2.ppf(0.01, 1), chi2.ppf(0.99, 1))
    for y in [smoothDis, smoothDon, texturedDis, texturedDon, curvedDis, curvedDon]:
        continue

