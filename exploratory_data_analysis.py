import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


df = np.array(pd.read_pickle('../data/diabetes_normalized_balanced.pickle'))

feature_size = df.shape[1]
ax = plt.axes()

plt.boxplot(df[:, :-1], positions=range(1, 9), widths=0.6, vert=False)

ax.set_yticklabels(['Pregnancies',
                    'Glucose',
                    'Blood Pressure',
                    'Skin Thickness',
                    'Insulin',
                    'BMI',
                    'Pedigree',
                    'Age'])

plt.show()
plt.savefig('../pic/boxPlot.png')
