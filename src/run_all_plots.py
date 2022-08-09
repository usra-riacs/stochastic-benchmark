import os

SIZES = [30,40,50,60,80,100,140]
ALPHAS = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
STATS = ['10','50','90']
for SIZE in SIZES:
    for ALPHA in ALPHAS:
        for STAT in STATS:
            print(SIZE,ALPHA,STAT)
            os.system('python3 generate_ws_dataframe_wishart.py ' + str(SIZE) + ' ' + str(ALPHA) + ' ' + str(STAT))  
