import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
import seaborn as sns


landmark_data = pd.read_csv('../salience_csv/Landmarken_mit_Faktorenwerten.csv', sep=";", decimal=",")
ids = landmark_data["ID"].to_list()
salience = landmark_data["Salienz_gerundet"].to_numpy()
category = landmark_data["Category"].to_numpy()
category_detailed = landmark_data["Category_detailed"].to_numpy()

relevant_ids = []
for file in os.listdir('../LM_Images_downscaled'):
    if file.endswith('.jpeg'):
        lm_id = '.'.join(file.split('.')[:2])
        if lm_id in ids and lm_id not in relevant_ids:
            relevant_ids.append(lm_id)

relevant_categories = []
relevant_categories_detailed = {}
for idx, value in enumerate(ids):
    if value in relevant_ids:
        relevant_categories.append(category[idx])
        if category[idx] not in list(relevant_categories_detailed.keys()):
            relevant_categories_detailed[category[idx]] = {}
            relevant_categories_detailed[category[idx]][category_detailed[idx]] = 1
        else:
            if category_detailed[idx] in list(relevant_categories_detailed[category[idx]].keys()):
                relevant_categories_detailed[category[idx]][category_detailed[idx]] = relevant_categories_detailed[category[idx]][category_detailed[idx]] + 1
            else:
                relevant_categories_detailed[category[idx]][category_detailed[idx]] = 1
        #print(relevant_categories_detailed)

relevant_categories_counter = Counter(relevant_categories)
print(relevant_categories_counter)
print(relevant_categories_detailed)

### bar general data ###
'''
fig, ax = plt.subplots(figsize =(8, 5))
category_labels = list(relevant_categories_counter.keys())
quantity_values = list(relevant_categories_counter.values())
ax.barh(category_labels, quantity_values)
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
ax.invert_yaxis()
for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y()+0.5,
             str(round((i.get_width()), 2)),
             fontsize = 10,
             color ='grey')
plt.xlabel('Anzahl')
plt.show()
'''

### detailed bar plot ###
'''
fig, ax = plt.subplots(figsize =(8, 4))

color_dict = {
    'func': 'red',
    'furn': 'blue',
    'info': 'green',
    'arch': 'purple',
    'shop': 'lavender'
}

x = []
y = []
colors = []

for cat in list(relevant_categories_detailed.keys()):
    for cat_detail in list(relevant_categories_detailed[cat].keys()):
        colors.append(color_dict[cat])
        x.append(cat_detail)
        y.append(relevant_categories_detailed[cat][cat_detail])

ax.bar(x, y, color=colors)

for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)

ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_tick_params(pad = 5, labelrotation=90)
ax.yaxis.set_tick_params(pad = 10)

ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)

#ax.invert_yaxis()
for i in ax.patches:
    plt.text(i.get_xy()[0]+0.1, i.get_height(),
             str(round((i.get_height()), 2)),
             fontsize = 9,
             color ='grey')
plt.ylabel('Anzahl')
plt.tight_layout()
plt.show()
'''
salience_door = []
for idx, id in enumerate(ids):
    if id in relevant_ids:
        if category_detailed[idx] == 'door':
            salience_door.append(salience[idx])
'''
sns.distplot(salience_door, hist=True, kde=True,
             bins=int(180/5), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.xlabel('Salienzwert')
plt.ylabel('Dichte')
'''

sns.distplot(salience_door, hist=True, kde=False,
             bins=int(180/5), color = 'darkblue',
             hist_kws={'edgecolor':'black'})
plt.xlabel('Salienzwert')
plt.ylabel('Häufigkeit')
plt.show()
