#!/bin/env python3

import deepmd_pybind

from dmff.utils import pair_buffer_scales, regularize_pairs
import jax.numpy as jnp
from dmff.admp.recip import generate_pme_recip, Ck_1
from dmff.admp.pme import energy_pme
from jax import jit, grad, jacfwd, jacrev, value_and_grad
from jax.scipy.special import erfc
import jaxopt
import os
import dpdata
from tqdm import tqdm
import freud
import numpy as np
import shutil
import jax

import dpdata
from typing import Tuple, Optional
import deepmd_pybind
import ase.io as IO

from ase.calculators.mixing import MixedCalculator, SumCalculator, LinearCombinationCalculator
from ase import units
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase.constraints import FixAtoms

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from ase.calculators.calculator import Calculator, all_changes, PropertyNotImplementedError
import time

qeq_total_time = []
dp_total_time = []
wall_total_time = []
calculate_call_time = []

qeq_charges = []
qeq_counter_list = []
qeq_energy, qeq_force = [], []

initial_charge_guess_list = []
initial_charge_guess_list.append(jnp.array(np.loadtxt("initial_charge_guess.txt")))
electrolyte_index = [1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440,1441,1442,1443,1444,1445,1446,1447,1448,1449,1450,1451,1452,1453,1454,1455,1456,1457,1458,1459,1460,1461,1462,1463,1464,1465,1466,1467,1468,1469,1470,1471,1472,1473,1474,1475,1476,1477,1478,1479,1480,1481,1482,1483,1484,1485,1486,1487,1488,1489,1490,1491,1492,1493,1494,1495,1496,1497,1498,1499,1500,1501,1502,1503,1504,1505,1506,1507,1508,1509,1510,1511,1512,1513,1514,1515,1516,1517,1518,1519,1520,1521,1522,1523,1524,1525,1526,1527,1528,1529,1530,1531,1532,1533,1534,1535,1536,1537,1538,1539,1540,1541,1542,1543,1544,1545,1546,1547,1548,1549,1550,1551,1552,1553,1554,1555,1556,1557,1558,1559,1560,1561,1562,1563,1564,1565,1566,1567,1568,1569,1570,1571,1572,1573,1574,1575,1576,1577,1578,1579,1580,1581,1582,1583,1584,1585,1586,1587,1588,1589,1590,1591,1592,1593,1594,1595,1596,1597,1598,1599,1600,1601,1602,1603,1604,1605,1606,1607,1608,1609,1610,1611,1612,1613,1614,1615,1616,1617,1618,1619,1620,1621,1622,1623,1624,1625,1626,1627,1628,1629,1630,1631,1632,1633,1634,1635,1636,1637,1638,1639,1640,1641,1642,1643,1644,1645,1646,1647,1648,1649,1650,1651,1652,1653,1654,1655,1656,1657,1658,1659,1660,1661,1662,1663,1664,1665,1666,1667,1668,1669,1670,1671,1672,1673,1674,1675,1676,1677,1678,1679,1680,1681,1682,1683,1684,1685,1686,1687,1688,1689,1690,1691,1692,1693,1694,1695,1696,1697,1698,1699,1700,1701,1702,1703,1704,1705,1706,1707,1708,1709,1710,1711,1712,1713,1714,1715,1716,1717,1718,1719,1720,1721,1722,1723,1724,1725,1726,1727,1728,1729,1730,1731,1732,1733,1734,1735,1736,1737,1738,1739,1740,1741,1742,1743,1744,1745,1746,1747,1748,1749,1750,1751,1752,1753,1754,1755,1756,1757,1758,1759,1760,1761,1762,1763,1764,1765,1766,1767,1768,1769,1770,1771,1772,1773,1774,1775,1776,1777,1778,1779,1780,1781,1782,1783,1784,1785,1786,1787,1788,1789,1790,1791,1792,1793,1794,1795,1796,1797,1798,1799,1800,1801,1802,1803,1804,1805,1806,1807,1808,1809,1810,1811,1812,1813,1814,1815,1816,1817,1818,1819,1820,1821,1822,1823,1824,1825,1826,1827,1828,1829,1830,1831,1832,1833,1834,1835,1836,1837,1838,1839,1840,1841,1842,1843,1844,1845,1846,1847,1848,1849,1850,1851,1852,1853,1854,1855,1856,1857,1858,1859,1860,1861,1862,1863,1864,1865,1866,1867,1868,1869,1870,1871,1872,1873,1874,1875,1876,1877,1878,1879,1880,1881,1882,1883,1884,1885,1886,1887,1888,1889,1890,1891,1892,1893,1894,1895,1896,1897,1898,1899,1900,1901,1902,1903,1904,1905,1906,1907,1908,1909,1910,1911,1912,1913,1914,1915,1916,1917,1918,1919,1920,1921,1922,1923,1924,1925,1926,1927,1928,1929,1930,1931,1932,1933,1934,1935,1936,1937,1938,1939,1940,1941,1942,1943,1944,1945,1946,1947,1948,1949,1950,1951,1952,1953,1954,1955,1956,1957,1958,1959,1960,1961,1962,1963,1964,1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039,2040,2041,2042,2043,2044,2045,2046,2047,2048,2049,2050,2051,2052,2053,2054,2055,2056,2057,2058,2059,2060,2061,2062,2063,2064,2065,2066,2067,2068,2069,2070,2071,2072,2073,2074,2075,2076,2077,2078,2079,2080,2081,2082,2083,2084,2085,2086,2087,2088,2089,2090,2091,2092,2093,2094,2095,2096,2097,2098,2099,2100,2101,2102,2103,2104,2105,2106,2107,2108,2109,2110,2111,2112,2113,2114,2115,2116,2117,2118,2119,2120,2121,2122,2123,2124,2125,2126,2127,2128,2129,2130,2131,2132,2133,2134,2135,2136,2137,2138,2139,2140,2141,2142,2143,2144,2145,2146,2147,2148,2149,2150,2151,2152,2153,2154,2155,2156,2157,2158,2159,2160,2161,2162,2163,2164,2165,2166,2167,2168,2169,2170,2171,2172,2173,2174,2175,2176,2177,2178,2179,2180,2181,2182,2183,2184,2185,2186,2187,2188,2189,2190,2191,2192,2193,2194,2195,2196,2197,2198,2199,2200,2201,2202,2203,2204,2205,2206,2207,2208,2209,2210,2211,2212,2213,2214,2215,2216,2217,2218,2219,2220,2221,2222,2223,2224,2225,2226,2227,2228,2229,2230,2231,2232,2233,2234,2235,2236,2237,2238,2239,2240,2241,2242,2243,2244,2245,2246,2247,2248,2249,2250,2251,2252,2253,2254,2255,2256,2257,2258,2259,2260,2261,2262,2263,2264,2265,2266,2267,2268,2269,2270,2271,2272,2273,2274,2275,2276,2277,2278,2279,2280,2281,2282,2283,2284,2285,2286,2287,2288,2289,2290,2291,2292,2293,2294,2295,2296,2297,2298,2299,2300,2301,2302,2303,2304,2305,2306,2307,2308,2309,2310,2311,2312,2313,2314,2315,2316,2317,2318,2319,2320,2321,2322,2323,2324,2325,2326,2327,2328,2329,2330,2331,2332,2333,2334,2335,2336,2337,2338,2339,2340,2341,2342,2343,2344,2345,2346,2347,2348,2349,2350,2351,2352,2353,2354,2355,2356,2357,2358,2359,2360,2361,2362,2363,2364,2365,2366,2367,2368,2369,2370,2371,2372,2373,2374,2375,2376,2377,2378,2379,2380,2381,2382,2383,2384,2385,2386,2387,2388,2389,2390,2391,2392,2393,2394,2395,2396,2397,2398,2399,2400,2401,2402,2403,2404,2405,2406,2407,2408,2409,2410,2411,2412,2413,2414,2415,2416,2417,2418,2419,2420,2421,2422,2423,2424,2425,2426,2427,2428,2429,2430,2431,2432,2433,2434,2435,2436,2437,2438,2439,2440,2441,2442,2443,2444,2445,2446,2447,2448,2449,2450,2451,2452,2453,2454,2455,2456,2457,2458,2459,2460,2461,2462,2463,2464,2465,2466,2467,2468,2469,2470,2471,2472,2473,2474,2475,2476,2477,2478,2479,2480,2481,2482,2483,2484,2485,2486,2487,2488,2489,2490,2491,2492,2493,2494,2495,2496,2497,2498,2499,2500,2501,2502,2503,2504,2505,2506,2507,2508,2509,2510,2511,2512,2513,2514,2515,2516,2517,2518,2519,2520,2521,2522,2523,2524,2525,2526,2527,2528,2529,2530,2531,2532,2533,2534,2535,2536,2537,2538,2539,2540,2541,2542,2543,2544,2545,2546,2547,2548,2549,2550,2551,2552,2553,2554,2555,2556,2557,2558,2559,2560,2561,2562,2563,2564,2565,2566,2567,2568,2569,2570,2571,2572,2573,2574,2575,2576,2577,2578,2579,2580,2581,2582,2583,2584,2585,2586,2587,2588,2589,2590,2591,2592,2593,2594,2595,2596,2597,2598,2599,2600,2601,2602,2603,2604,2605,2606,2607,2608,2609,2610,2611,2612,2613,2614,2615,2616,2617,2618,2619,2620,2621,2622,2623,2624,2625,2626,2627,2628,2629,2630,2631,2632,2633,2634,2635,2636,2637,2638,2639,2640,2641,2642,2643,2644,2645,2646,2647,2648,2649,2650,2651,2652,2653,2654,2655,2656,2657,2658,2659,2660,2661,2662,2663,2664,2665,2666,2667,2668,2669,2670,2671,2672,2673,2674,2675,2676,2677,2678,2679,2680,2681,2682,2683,2684,2685,2686,2687,2688,2689,2690,2691,2692,2693,2694,2695,2696,2697,2698,2699,2700,2701,2702,2703,2704,2705,2706,2707,2708,2709,2710,2711,2712,2713,2714,2715,2716,2717,2718,2719,2720,2721,2722,2723,2724,2725,2726,2727,2728,2729,2730,2731,2732,2733,2734,2735,2736,2737,2738,2739,2740,2741,2742,2743,2744,2745,2746,2747,2748,2749,2750,2751,2752,2753,2754,2755,2756,2757,2758,2759,2760,2761,2762,2763,2764,2765,2766,2767,2768,2769,2770,2771,2772,2773,2774,2775,2776,2777,2778,2779,2780,2781,2782,2783,2784,2785,2786,2787,2788,2789,2790,2791,2792,2793,2794,2795,2796,2797,2798,2799,2800,2801,2802,2803,2804,2805,2806,2807,2808,2809,2810,2811,2812,2813,2814,2815,2816,2817,2818,2819,2820,2821,2822,2823,2824,2825,2826,2827,2828,2829,2830,2831,2832,2833,2834,2835,2836,2837,2838,2839,2840,2841,2842,2843,2844,2845,2846,2847,2848,2849,2850,2851,2852,2853,2854,2855,2856,2857,2858,2859,2860,2861,2862,2863,2864,2865,2866,2867,2868,2869,2870,2871,2872,2873,2874,2875,2876,2877,2878,2879,2880,2881,2882,2883,2884,2885,2886,2887,2888,2889,2890,2891,2892,2893,2894,2895,2896,2897,2898,2899,2900,2901,2902,2903,2904,2905,2906,2907,2908,2909,2910,2911,2912,2913,2914,2915,2916,2917,2918,2919,2920,2921,2922,2923,2924,2925,2926,2927,2928,2929,2930,2931,2932,2933,2934,2935,2936,2937,2938,2939,2940,2941,2942,2943,2944,2945,2946,2947,2948,2949,2950,2951,2952,2953,2954,2955,2956,2957,2958,2959,2960,2961,2962,2963,2964,2965,2966,2967,2968,2969,2970,2971,2972,2973,2974,2975,2976,2977,2978,2979,2980,2981,2982,2983,2984,2985,2986,2987,2988,2989,2990,2991,2992,2993,2994,2995,2996,2997,2998,2999,3000,3001,3002,3003,3004,3005,3006,3007,3008,3009,3010,3011,3012,3013,3014,3015,3016,3017,3018,3019,3020,3021,3022,3023,3024,3025,3026,3027,3028,3029,3030,3031,3032,3033,3034,3035,3036,3037,3038,3039,3040,3041,3042,3043,3044,3045,3046,3047,3048,3049,3050,3051,3052,3053,3054,3055,3056,3057,3058,3059,3060,3061,3062,3063,3064,3065,3066,3067,3068,3069]



most_bottum_Li_index = [2,12,21,30,38,47,54,63,70,78,85,95,104,114,122,131,138,146,154,163,171,179,187,197,205,215,223,233,242,251,258,267,274,285,294,304,312,321,328,337,345,355,363,372,381,391,399,409,417,427,434,444,452,461,470,479,487,496,504,514,522,531,538,548,556,565,572,582,590,600,607,616,623,633,641,652,660,669,676,686,693,704,712,722,730,740,748,758,766,776,785,793,801,810,818,828,837,846,854,863]

most_upper_Li_index = [898,903,907,913,918,924,930,936,942,949,955,960,964,969,974,980,986,993,998,1004,1009,1016,1021,1026,1031,1036,1041,1046,1050,1056,1062,1068,1074,1078,1082,1087,1092,1098,1104,1110,1115,1120,1125,1131,1135,1140,1145,1150,1155,1160,1166,1171,1176,1182,1186,1192,1197,1203,1208,1213,1218,1224,1230,1235,1240,1246,1252,1257,1262,1267,1273,1279,1285,1290,1295,1299,1304,1310,1316,1321,1327,1331,1336,1341,1346,1351,1356,1361,1366,1371,1375,1382,1387,1393,1398,1403,1407,1413,1418,1424]

fix_Li_bottum = [2, 3, 12, 13, 21, 22, 30, 31, 38, 39, 47, 48, 54, 55, 63, 64, 70, 71, 78, 79, 85, 86, 95, 96, 104, 105, 114, 115, 122, 123, 131, 132, 138, 139, 146, 147, 154, 155, 163, 164, 171, 172, 179, 180, 187, 188, 197, 198, 205, 206, 215, 216, 223, 224, 233, 234, 242, 243, 251, 252, 258, 259, 267, 268, 274, 275, 285, 286, 294, 295, 304, 305, 312, 313, 321, 322, 328, 329, 337, 338, 345, 346, 355, 356, 363, 364, 372, 373, 381, 382, 391, 392, 399, 400, 409, 410, 417, 418, 427, 428, 434, 435, 444, 445, 452, 453, 461, 462, 470, 471, 479, 480, 487, 488, 496, 497, 504, 505, 514, 515, 522, 523, 531, 532, 538, 539, 548, 549, 556, 557, 565, 566, 572, 573, 582, 583, 590, 591, 600, 601, 607, 608, 616, 617, 623, 624, 633, 634, 641, 642, 652, 653, 660, 661, 669, 670, 676, 677, 686, 687, 693, 694, 704, 705, 712, 713, 722, 723, 730, 731, 740, 741, 748, 749, 758, 759, 766, 767, 776, 777, 785, 786, 793, 794, 801, 802, 810, 811, 818, 819, 828, 829, 837, 838, 846, 847, 854, 855, 863, 864]

fix_Li_upper = [897, 898, 902, 903, 906, 907, 912, 913, 917, 918, 923, 924, 929, 930, 935, 936, 941, 942, 948, 949, 954, 955, 959, 960, 963, 964, 968, 969, 973, 974, 979, 980, 985, 986, 992, 993, 997, 998, 1003, 1004, 1008, 1009, 1015, 1016, 1020, 1021, 1025, 1026, 1030, 1031, 1035, 1036, 1040, 1041, 1045, 1046, 1049, 1050, 1055, 1056, 1061, 1062, 1067, 1068, 1073, 1074, 1077, 1078, 1081, 1082, 1086, 1087, 1091, 1092, 1097, 1098, 1103, 1104, 1109, 1110, 1114, 1115, 1119, 1120, 1124, 1125, 1130, 1131, 1134, 1135, 1139, 1140, 1144, 1145, 1149, 1150, 1154, 1155, 1159, 1160, 1165, 1166, 1170, 1171, 1175, 1176, 1181, 1182, 1185, 1186, 1191, 1192, 1196, 1197, 1202, 1203, 1207, 1208, 1212, 1213, 1217, 1218, 1223, 1224, 1229, 1230, 1234, 1235, 1239, 1240, 1245, 1246, 1251, 1252, 1256, 1257, 1261, 1262, 1266, 1267, 1272, 1273, 1278, 1279, 1284, 1285, 1289, 1290, 1294, 1295, 1298, 1299, 1303, 1304, 1309, 1310, 1315, 1316, 1320, 1321, 1326, 1327, 1330, 1331, 1335, 1336, 1340, 1341, 1345, 1346, 1350, 1351, 1355, 1356, 1360, 1361, 1365, 1366, 1370, 1371, 1374, 1375, 1381, 1382, 1386, 1387, 1392, 1393, 1397, 1398, 1402, 1403, 1406, 1407, 1412, 1413, 1417, 1418, 1423, 1424]


def calculate_model_devi_v(vs):
    vs_devi = np.std(vs, axis=0)
    max_devi_v = np.max(vs_devi, axis=-1)
    min_devi_v = np.min(vs_devi, axis=-1)
    avg_devi_v = np.linalg.norm(vs_devi, axis=-1) / 3
    return max_devi_v, min_devi_v, avg_devi_v

def calculate_model_devi_f(fs):
    fs_devi = np.linalg.norm(np.std(fs, axis=0), axis=-1)
    max_devi_f = np.max(fs_devi, axis=-1)
    min_devi_f = np.min(fs_devi, axis=-1)
    avg_devi_f = np.mean(fs_devi, axis=-1)
    return max_devi_f, min_devi_f, avg_devi_f

def calculate_model_devi_e(es):
    es_devi = np.std(es, axis=0)
    es_devi = np.squeeze(es_devi, axis=-1)
    return es_devi

def write_model_devi_out(data, fname):
    header = "%10s" % ("step")
    for item in "vf":
        header += "%19s%19s%19s" % (f"max_devi_{item}", f"min_devi_{item}", f"avg_devi_{item}")
    header += "%19s" % "devi_e"

    with open(fname, "ab") as fp:
        np.savetxt(fp, data, fmt=["%12d"] + ["%19.6e" for _ in range(7)], delimiter="", header=header)

def calculate_model_devi_cpp(
        file_name: str = "wrapped_trajectory.traj",
        file_type: str = "ase/traj",
        type_dict: dict = {"Li": 0, "C": 1, "H": 2, "O": 3, "P": 4, "F": 5},
        pb_file: list = ["graph.000.pb", "graph.001.pb", "graph.002.pb", "graph.003.pb"],
        frequency: int = 1,     
):

    graphs = [deepmd_pybind.DeepPot(tmp) for tmp in pb_file]
    if file_type == "ase/traj":
        atoms = IO.read(file_name, index=":")
        nframes = len(atoms)
        iterator = tqdm(range(0, nframes, 1))
        devi = []
        for iframe in iterator:
            result = []
            result_data = []
            coordinate = atoms[iframe].get_positions().reshape([1, -1])[0]
            cell = atoms[iframe].get_cell().reshape([1, -1])[0]
            symbols = atoms[iframe].get_chemical_symbols()
            atype = [type_dict[tmp] for tmp in symbols]
            energies, forces, virials = [], [], []
            for tmp in graphs:
                e, f, v = tmp.compute(coordinate, atype, cell)
                energies.append(e/atoms[iframe].get_global_number_of_atoms())
                forces.append(np.array(f).reshape([-1, 3]))
                virials.append(np.array(v)/atoms[iframe].get_global_number_of_atoms())
            energies, forces, virials = np.array(energies), np.array(forces), np.array(virials)
            #print(list(calculate_model_devi_e(energies)))
            result.append(iframe*frequency)
            result_data.extend(list(calculate_model_devi_v(virials)))
            result_data.extend(list(calculate_model_devi_f(forces)))
            result_data.extend([calculate_model_devi_e(energies),])
            result_data = np.vstack(result_data).T
            result.extend(result_data[0])
            devi.append(result)
        write_model_devi_out(devi, "model_devi.out")


class NeighborListFreud_numpy:
    def __init__(self, box, rcut, cov_map, padding=True, max_shape=0):
        if freud is None:
            raise ImportError("Freud not installed.")
        self.fbox = freud.box.Box.from_matrix(box)
        self.rcut = rcut
        self.capacity_multiplier = None
        self.padding = padding
        self.cov_map = cov_map
        self.max_shape = max_shape
    
    def _do_cov_map(self, pairs):
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        pairs = np.concatenate([pairs, nbond[:, None]], axis=1)
        return pairs

    def allocate(self, coords, box=None):
        self._positions = coords  # cache it
        fbox = freud.box.Box.from_matrix(box) if box is not None else self.fbox
        aq = freud.locality.AABBQuery(fbox, coords)
        res = aq.query(coords, dict(r_max=self.rcut, exclude_ii=True))
        nlist = res.toNeighborList()
        nlist = np.vstack((nlist[:, 0], nlist[:, 1])).T
        nlist = nlist.astype(np.int32)
        msk = (nlist[:, 0] - nlist[:, 1]) < 0
        nlist = nlist[msk]
        if self.capacity_multiplier is None:
            if self.max_shape == 0:
                self.capacity_multiplier = int(nlist.shape[0] * 1.5)
            else:
                self.capacity_multiplier = self.max_shape
        
        if not self.padding:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs

        if self.max_shape == 0:
            self.capacity_multiplier = max(self.capacity_multiplier, nlist.shape[0])
        else:
            self.capacity_multiplier = self.max_shape

        padding_width = self.capacity_multiplier - nlist.shape[0]
        if padding_width == 0:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        elif padding_width > 0:
            padding = np.ones((self.capacity_multiplier - nlist.shape[0], 2), dtype=np.int32) * coords.shape[0]
            nlist = np.vstack((nlist, padding))
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        else:
            raise ValueError("padding width < 0")

    def update(self, positions, box=None):
        self.allocate(positions, box)

    @property
    def pairs(self):
        return self._pairs

    @property
    def scaled_pairs(self):
        return self._pairs

    @property
    def positions(self):
        return self._positions

def get_neighbor_list_numpy(box, rc, positions, natoms, padding=True, max_shape=0):
    nbl = NeighborListFreud_numpy(box, rc, np.zeros([natoms, natoms], dtype=np.int32), padding=padding, max_shape=max_shape)
    nbl.allocate(positions)
    pairs = nbl.pairs
    #pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
    return pairs

def determine_chi(
        box,
        positions: np.ndarray,
        symbols: list,
        mode: int=1,  # 1= constant charge, 2=constant potential
        most_upper_index: list=most_upper_Li_index,
        most_bottum_index: list=most_bottum_Li_index,
        bottum_external_chi: float=0.0,
        upper_external_chi: float=0.0,
):
    
    chi_0 = np.array([name2chi[tmp] for tmp in symbols])
    if mode == 1:
        return chi_0
    elif mode == 2:
        natoms = len(symbols)
        coord_number = np.zeros(natoms)
        chi_t = np.zeros(natoms)

        pairs = get_neighbor_list_numpy(box, 3.5, positions, natoms, padding=False, max_shape=0)

        for i, j, k in pairs:
            if symbols[i] == "Li" and symbols[j] == "Li":
                coord_number[i] += 1
                coord_number[j] += 1

        black_Li, green_Li = [], []

        Li_bottum_index, Li_upper_index = [], []
        Li_all_index = []
        for iatom, symbol in enumerate(symbols):
            if symbol == "Li":
                y_coord = positions[iatom, 1]
                if coord_number[iatom] >= 8.0:
                    if y_coord < box[1,1]/2:
                        chi_t[iatom] = bottum_external_chi
                        Li_bottum_index.append(iatom)
                        black_Li.append(iatom)
                    elif y_coord > box[1,1]/2:
                        chi_t[iatom] = upper_external_chi
                        Li_upper_index.append(iatom)
                        black_Li.append(iatom)
                else:
                    green_Li.append(iatom)

        for idx in most_bottum_index:
            chi_t[idx] = bottum_external_chi
            if idx not in Li_bottum_index:
                Li_bottum_index.append(idx)
                black_Li.append(idx)

        for idx in most_upper_index:
            chi_t[idx] = upper_external_chi
            if idx not in Li_upper_index:
                Li_upper_index.append(idx)
                black_Li.append(idx)

        #with open("Li_bottum_index.txt", "a") as f:
        #    f.write(" ".join(map(str, Li_bottum_index)))
        #    f.write("\n")
        #with open("Li_upper_index.txt", "a") as f:
        #    f.write(" ".join(map(str, Li_upper_index)))
        #    f.write("\n")
        return chi_0+chi_t, 0, 0


class NeighborListFreud:
    def __init__(self, box, rcut, cov_map, padding=True, max_shape=0):
        if freud is None:
            raise ImportError("Freud not installed.")
        self.fbox = freud.box.Box.from_matrix(box)
        self.rcut = rcut
        self.capacity_multiplier = None
        self.padding = padding
        self.cov_map = cov_map
        self.max_shape = max_shape
    
    def _do_cov_map(self, pairs):
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        pairs = jnp.concatenate([pairs, nbond[:, None]], axis=1)
        return pairs

    def allocate(self, coords, box=None):
        self._positions = coords  # cache it
        fbox = freud.box.Box.from_matrix(box) if box is not None else self.fbox
        aq = freud.locality.AABBQuery(fbox, coords)
        res = aq.query(coords, dict(r_max=self.rcut, exclude_ii=True))
        nlist = res.toNeighborList()
        nlist = np.vstack((nlist[:, 0], nlist[:, 1])).T
        nlist = nlist.astype(np.int32)
        msk = (nlist[:, 0] - nlist[:, 1]) < 0
        nlist = nlist[msk]
        if self.capacity_multiplier is None:
            if self.max_shape == 0:
                self.capacity_multiplier = int(nlist.shape[0] * 1.3)
            else:
                self.capacity_multiplier = self.max_shape
        
        if not self.padding:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs

        if self.max_shape == 0:
            self.capacity_multiplier = max(self.capacity_multiplier, nlist.shape[0])
        else:
            self.capacity_multiplier = self.max_shape

        padding_width = self.capacity_multiplier - nlist.shape[0]
        if padding_width == 0:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        elif padding_width > 0:
            padding = np.ones((self.capacity_multiplier - nlist.shape[0], 2), dtype=np.int32) * coords.shape[0]
            nlist = np.vstack((nlist, padding))
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        else:
            raise ValueError("padding width < 0")

    def update(self, positions, box=None):
        self.allocate(positions, box)

    @property
    def pairs(self):
        return self._pairs

    @property
    def scaled_pairs(self):
        return self._pairs

    @property
    def positions(self):
        return self._positions

@jit
def ds_pairs(positions, box, pairs):
    pos1 = positions[pairs[:,0].astype(int)]
    pos2 = positions[pairs[:,1].astype(int)]
    box_inv = jnp.linalg.inv(box)
    dpos = pos1 - pos2
    dpos = dpos.dot(box_inv)
    dpos -= jnp.floor(dpos+0.5)
    dr = dpos.dot(box)
    ds = jnp.linalg.norm(dr,axis=1)
    return ds

def typemap_list_to_symbols(atom_numbs: list, atom_names: list):
    atomic_symbols = []
    idx = 0
    for numb in atom_numbs:
        atomic_symbols.extend((atom_names[idx], )*numb)
        idx += 1
    return atomic_symbols

def generate_get_energy(kappa, K1, K2, K3):
    pme_recip_fn = generate_pme_recip(
        Ck_fn=Ck_1,
        kappa=kappa / 10,
        gamma=False,
        pme_order=6,
        K1=K1,
        K2=K2,
        K3=K3,
        lmax=0,
    )
    def get_energy_kernel(positions, box, pairs, charges, mscales):
        atomCharges = charges
        atomChargesT = jnp.reshape(atomCharges, (-1, 1))
        return energy_pme(
            positions * 10,
            box * 10,
            pairs,
            atomChargesT,
            None,
            None,
            None,
            mscales,
            None,
            None,
            None,
            pme_recip_fn,
            kappa / 10,
            K1,
            K2,
            K3,
            0,
            False,
        )
    def get_energy(positions, box, pairs, charges, mscales):
        return get_energy_kernel(positions, box, pairs, charges, mscales)
    return get_energy

@jit 
def get_Energy_Qeq_2(charges, positions, box, pairs, eta, chi, hardness):
    @jit 
    def get_Energy_PME():
        pme = generate_get_energy(4.3804348, 45, 123, 22)
        e = pme(positions/10, box/10, pairs, charges, mscales=jnp.array([1., 1., 1., 1., 1., 1.]))
        return e
    @jit 
    def get_Energy_Correction():
        ds = ds_pairs(positions, box, pairs)
        buffer_scales = pair_buffer_scales(pairs)
        e_corr_pair = charges[pairs[:,0]] * charges[pairs[:,1]] * erfc(ds / (jnp.sqrt(2) * jnp.sqrt(eta[pairs[:,0]]**2 + eta[pairs[:,1]]**2))) * 1389.35455846 / ds  * buffer_scales
        e_corr_self = charges * charges * 1389.35455846 /(2*jnp.sqrt(jnp.pi) * eta)
        return  -jnp.sum(e_corr_pair) + jnp.sum(e_corr_self)
    @jit
    def get_Energy_Onsite():
        E_tf =  (chi * charges + 0.5 * hardness * charges *charges) * 96.4869
        #E_tf = 0.5 * hardness * charges *charges * 96.4869
        return jnp.sum(E_tf)
    @jit 
    def get_dipole_correction():
        V = jnp.linalg.det(box)
        pre_corr = 2 * jnp.pi / V * 1389.35455846
        Mz = jnp.sum(charges * positions[:, 1])
        e_corr = pre_corr * Mz**2       
        return jnp.sum(e_corr)

    return (get_Energy_PME() + get_Energy_Correction() + get_Energy_Onsite() + get_dipole_correction()) / 96.4869

def fn_value_and_proj_grad(func, constraint_matrix, has_aux=False):
    def value_and_proj_grad(*arg, **kwargs):
        value, grad = jax.value_and_grad(func, has_aux=has_aux)(*arg, **kwargs)
        # n * 1
        a = jnp.matmul(constraint_matrix, grad.reshape(-1, 1))
        # n * 1
        b = jnp.sum(constraint_matrix * constraint_matrix, axis=1, keepdims=True)
        # 1 * N
        delta_grad = jnp.matmul((a / b).T, constraint_matrix)
        # N
        proj_grad = grad - delta_grad.reshape(-1)
        return value, proj_grad
    return value_and_proj_grad

@jit
def solve_q_pg(charges, positions, box, pairs, eta, chi, hardness):
    func = fn_value_and_proj_grad(get_Energy_Qeq_2, jnp.ones_like(charges).reshape(1, -1))
    solver = jaxopt.LBFGS(
        fun=func,
        value_and_grad=True,
        tol=1e-2,
        )
    res = solver.run(charges, positions, box, pairs, eta, chi, hardness)
    x_opt = res.params
    return x_opt

@jit
def get_force(charges, positions, box, pairs, eta, chi, hardness):
    energy,force = value_and_grad(get_Energy_Qeq_2,argnums=(1))(charges, positions, box, pairs, eta, chi, hardness)
    return energy, -force

def get_qeq_energy_and_force_pg(charges, positions, box, pairs, eta, chi, hardness):
    q = solve_q_pg(charges, positions, box, pairs, eta, chi, hardness)
    energy, force = get_force(q, positions, box, pairs, eta, chi, hardness)
    return energy, force, q

def get_neighbor_list(box, rc, positions, natoms, padding=True, max_shape=0):
    nbl = NeighborListFreud(box, rc, jnp.zeros([natoms, natoms], dtype=jnp.int32), padding=padding, max_shape=max_shape)
    nbl.allocate(positions)
    pairs = nbl.pairs
    pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
    return pairs



if TYPE_CHECKING:
    from ase import Atoms

__all__ = ["QEQ"]

name2eta = {
    "Li":10.0241,
    "C":7.0000,
    "H":7.4366,
    "O":8.9989,
    "P":7.0946,
    "F":8.0000,
}

name2chi = {
    "Li":-3.0000,
    "C":5.8678,
    "H":5.3200,
    "O":8.5000,
    "P":1.8000,
    "F":9.0000,
}

name2index = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Uut": 113,
    "Uuq": 114,
    "Uup": 115,
    "Uuh": 116,
    "Uus": 117,
    "Uuo": 118,
}

R_Covalence = (
    2.0,  # ghost?
    0.31,  #H
    0.28,  #He
    1.28,  #Li
    0.96,  #Be
    0.84,  #B
    0.76,  #C
    0.71,  #N
    0.66,  #O
    0.57,  #F
    0.58,  #Ne
    1.66,  #Na
    1.41,  #Mg
    1.21,  #Al
    1.11,  #Si
    1.07,  #P
    1.05,  #S
    1.02,  #Cl
    1.06,  #Ar
    2.03,  #K
    1.76,  #Ca
    1.70,  #Sc
    1.60,  #Ti
    1.53,  #V
    1.39,  #Cr
    1.61,  #Mn
    1.52,  #Fe
    1.50,  #Co
    1.24,  #Ni
    1.32,  #Cu
    1.22,  #Zn
    1.22,  #Ga
    1.20,  #Ge
    1.19,  #As
    1.20,  #Se
    1.20,  #Br
    1.16,  #Kr
    2.20,  #Rb
    1.95,  #Sr
    1.90,  #Y
    1.75,  #Zr
    1.64,  #Nb
    1.54,  #Mo
    1.47,  #Tc
    1.46,  #Ru
    1.42,  #Rh
    1.39,  #Pd
    1.45,  #Ag
    1.44,  #Cd
    1.42,  #In
    1.39,  #Sn
    1.39,  #Sb
    1.38,  #Te
    1.39,  #I
    1.40,  #Xe
    2.44,  #Cs
    2.15,  #Ba
    2.07,  #La
    2.04,  #Ce
    2.03,  #Pr
    2.01,  #Nd
    1.99,  #Pm
    1.98,  #Sm
    1.98,  #Eu
    1.96,  #Gd
    1.94,  #Tb
    1.92,  #Dy
    1.92,  #Ho
    1.89,  #Er
    1.90,  #Tm
    1.87,  #Yb
    1.87,  #Lu
    1.75,  #Hf
    1.70,  #Ta
    1.62,  #W
    1.51,  #Re
    1.44,  #Os
    1.41,  #Ir
    1.36,  #Pt
    1.36,  #Au
    1.32,  #Hg
    1.45,  #Tl
    1.46,  #Pb
    1.48,  #Bi
    1.40,  #Po
    1.50,  #At
    1.50,  #Rn
    2.60,  #Fr
    2.21,  #Ra
    2.15,  #Ac
    2.06,  #Th
    2.00,  #Pa
    1.96,  #U
    1.90,  #Np
    1.87,  #Pu
    1.80,  #Am
    1.69  #Cm
)


def cell_to_box(a, b, c, alpha, beta, gamma):
    alpha = alpha / 180 * np.pi
    beta  = beta  / 180 * np.pi
    gamma = gamma / 180 * np.pi

    box = np.zeros((3,3), dtype=np.double) 
    box[0, 0] = a
    box[0, 1] = 0
    box[0, 2] = 0
    box[1, 0] = b * np.cos(gamma)
    box[1, 1] = b * np.sin(gamma)
    box[1, 2] = 0
    box[2, 0] = c * np.cos(beta)
    box[2, 1] = c * (np.cos(alpha)-np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
    box[2, 2] = c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma))**2)
    return box

def apply_wall_2(positions, coord, epsilon, sigma, cutoff, dim, side, index):
    coeff1 = 48.0 * epsilon * sigma**12
    coeff2 = 24.0 * epsilon * sigma**6
    natoms = positions.shape[0]
    forces = np.zeros_like(positions)

    for idx in index:
        if side < 0:
            delta = positions[idx, dim] - coord
        else:
            delta = coord - positions[idx, dim]
        
        if delta > cutoff:
            continue
        rinv = 1.0/delta
        r2inv = rinv*rinv
        r6inv = r2inv*r2inv*r2inv
        fwall = side * r6inv*(coeff1*r6inv - coeff2) * rinv
        forces[idx, dim] = fwall
    return forces

class DPQEQ(Calculator):
    name = "DPQEQ"
    implemented_properties = ["energy", "forces"]

    def __init__(self, 
                model: Union[str, "Path"],
                label: str = "DPQEQ",
                pairs_max_shape: int=0,
                type_map: Optional[Dict[str, int]] = None,
                mode: int = 0, # DPQEQ, 1=only DP, 2=only qeq
                const_potential: bool = True, # constant charge  
                voltage: float = 0.0, # constant potential voltage
                **kwargs
            ) -> None:
        Calculator.__init__(self, label=label, **kwargs)

        #self.dp = DeepPotential(str(Path(model).resolve()))

        self.dp = deepmd_pybind.DeepPot(str(Path(model).resolve()))
        self.pairs_max_shape = pairs_max_shape 
        self.mode = mode    
        self.const_potential = const_potential
        self.voltage = voltage
        self.type_map = type_map

    def calculate(
        self,
        atoms: Optional["Atoms"] = None,
        properties: List[str] = ["energy", "forces"],
        system_changes: List[str] = all_changes,
        ):
        if atoms is not None:
            self.atoms = atoms.copy()

        positions = jnp.array(self.atoms.get_positions())
        cell = self.atoms.get_cell_lengths_and_angles()
        box  = jnp.array(cell_to_box(cell[0], cell[1], cell[2], cell[3], cell[4], cell[5]))
        symbols = self.atoms.get_chemical_symbols()
        charges = jnp.array(np.random.random(len(symbols))) 
        if self.const_potential is False:
            chi = jnp.array([name2chi[tmp] for tmp in symbols])
        else:
            chi, z0, z1 = determine_chi(np.array(box), np.array(positions), symbols, 2, most_upper_Li_index,  most_bottum_Li_index, bottum_external_chi=6.0, upper_external_chi=-2.0)

        hardness = jnp.array([name2eta[tmp] for tmp in symbols])
        eta = jnp.array([R_Covalence[name2index[tmp]] for tmp in symbols])
        natoms = self.atoms.get_global_number_of_atoms()

        qeq_counter = int(np.sum(qeq_counter_list))
        
        coord = self.atoms.get_positions().reshape([1, -1])[0]
        symbols = self.atoms.get_chemical_symbols()
        cell = self.atoms.get_cell().reshape([1, -1])[0]
        atype = [self.type_map[tmp] for tmp in symbols]
        
        time1 = time.process_time()
        e, f, v = self.dp.compute(coord, atype, cell) # dp cpp api infer for effeciency
        time2 = time.process_time()
        print("DP infer costs time: %.3f"%(time2-time1))

        if self.mode == 0:
            if qeq_counter%3 != 0:
                energy = qeq_energy[-1]
                force  = qeq_force[-1]
            else:            
                time1 = time.process_time()
                pairs = get_neighbor_list(box, 6, positions, natoms, padding=True, max_shape=self.pairs_max_shape)
                charge = initial_charge_guess_list[-1]
                energy, force, charge = get_qeq_energy_and_force_pg(charge, positions, box, pairs, eta, chi, hardness)
                initial_charge_guess_list.append(charge)
                del(initial_charge_guess_list[0])
                qeq_charges.append(charge)
                qeq_energy.append(energy)
                qeq_force.append(force)
                time2 = time.process_time()
                print("QEq calculation costs time: %.3f"%(time2-time1))                        

            force_1 = apply_wall_2(atoms.get_positions(), 15.50,  0.025, 2.451, 2.5, 1,  1, electrolyte_index)    
            force_2 = apply_wall_2(atoms.get_positions(), 74.00,  0.025, 2.451, 2.5, 1, -1, electrolyte_index)    

            self.results["energy"] = np.array(energy) + e
            self.results["forces"] = np.array(force).reshape(-1, 3) + np.array(f).reshape(-1, 3) - force_1 - force_2

            qeq_counter_list.append(1)
            
def parse_lammps_input(file_name="input.lammps"):
    with open(file_name, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]

    for line in lines:
        if line.startswith("variable        NSTEPS"):
            nstep = float(line.split()[-1])
        elif line.startswith("variable        TEMP"):
            temp = float(line.split()[-1])
        elif line.startswith("variable        DUMP_FREQ"):
            dump_freq = int(line.split()[-1])
        elif line.startswith("timestep"):
            dt = float(line.split()[-1])
    return nstep, temp, dump_freq, dt

np.random.seed(2333)


def run_md_ase(input_file, potential_file, temp, dt, step, dump_freq):
    atoms = IO.read(input_file)
    calc2 = DPQEQ(model=potential_file, pairs_max_shape=200000, type_map={"Li":0, "C":1, "H":2, "O":3, "P":4, "F":5}, mode=0, const_potential=True, voltage=-12)
    atoms.calc = calc2

    constraint = FixAtoms(indices=fix_Li_bottum+fix_Li_upper)
    atoms.set_constraint([constraint,])

    T = temp
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    md = NPT(atoms, timestep=dt*1000*units.fs, temperature_K=T, ttime = 20*units.fs, pfactor = None)
    traj = Trajectory('md.traj', 'w', atoms)
    md.attach(MDLogger(md, atoms, 'md.log', header=True, stress=False, mode='w'), interval=dump_freq)
    md.attach(traj.write, interval=dump_freq)
    time1 = time.process_time()   
    md.run(int(step))
    time2 = time.process_time()
    print("it costs: %.5f s"%(time2 - time1))

def traj_wrap():
    trajectory = IO.read('md.traj', index=':')  
    for atoms in trajectory:
        atoms.wrap()  
    IO.write('wrapped_trajectory.traj', trajectory)

def convert_traj_to_lmp(frequency=1):
    total_atoms = IO.read("wrapped_trajectory.traj", index=":")
    nframes = len(total_atoms)

    for iframe in range(nframes):
        total_atoms[iframe].write("traj/%d.lammpstrj"%(iframe*frequency), format="lammps-data", specorder=["Li", "C", "H", "O", "P", "F"])

def get_qeq_charge_for_md_traj(file_name="wrapped_trajectory.traj", frequency=1):
    atoms = IO.read(file_name, index=":")
    nframes = len(atoms)
    iterator = tqdm(range(0, nframes, frequency))
    qeq_charges = []
    for iframe in iterator:
        coordinate = atoms[iframe].get_positions()
        cell_tmp = atoms[iframe].get_cell_lengths_and_angles()
        box = cell_to_box(cell_tmp[0], cell_tmp[1], cell_tmp[2], cell_tmp[3], cell_tmp[4], cell_tmp[5])
        symbols = atoms[iframe].get_chemical_symbols()
        eta = jnp.array([R_Covalence[name2index[tmp]] for tmp in symbols])
        chi, z0, z1 = determine_chi(np.array(box), np.array(coordinate), symbols, 2, most_upper_Li_index,  most_bottum_Li_index, bottum_external_chi=6.0, upper_external_chi=-2.0)
        hardness = jnp.array([name2eta[tmp] for tmp in symbols])
        pairs = get_neighbor_list(box, 6, coordinate, len(coordinate), padding=True, max_shape=200000)
        charge = initial_charge_guess_list[-1]
        energy, force, charge = get_qeq_energy_and_force_pg(charge, coordinate, box, pairs, eta, chi, hardness)
        qeq_charges.append(charge)
        initial_charge_guess_list.append(charge)
        del(initial_charge_guess_list[0])
    np.savetxt("qeq_charges", np.reshape(qeq_charges, [-1, 3070]), fmt="%.4f")

# everything are prepared, now run md simulations
# parse lammps input file 
nstep, temp, dump_freq, dt = parse_lammps_input()

# run MD
run_md_ase("POSCAR", "graph.000.pb", temp, dt, nstep, dump_freq)

# traj convert
traj_wrap()

# calculate model devi
#calculate_model_devi_cpp(frequency=dump_freq)

# export traj/*.lammpstrj
#convert_traj_to_lmp(frequency=dump_freq)

# save qeq charges for analysis
#np.savetxt("qeq_charges", np.array(qeq_charges)[::dump_freq, :], fmt="%.4f")
#get_qeq_charge_for_md_traj(frequency=1)
