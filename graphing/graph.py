import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
                '%.2f%%' % height,
                ha='center', va='bottom')

# color_normal = '#2C7BB6'
# color_adversarial = '#D7191C'

# accuracy_before = [
#             [99.14, 98.61, 96.83, 99.27, 96.96, 98.48, 95.21, 96.94, 95, 96.18],
#             [99.82, 95.67, 97.38, 98.89, 93.09, 94.47, 96.21, 95.37, 93.34, 98.25],
#             [99.75, 96.91, 91.75, 91.37, 97.3, 95.38, 96.26, 91.23, 95.54, 92.77],
#             [38.32, 35.96, 36.42, 38.5, 37.93, 40.39, 35.44, 42.41, 36.93, 35.73]
#         ]
# accuracy_after = [
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0.258, 0.102, 0.639, 0.335, 0.262, 0.146, 0.594, 0.689, 0.841, 1.2],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [2.144, 1.266, 2.091, 2.554, 1.703, 1.221, 2.128, 1.596, 1.041, 1.158]
#         ]

# ticks = ['Standard', 'More_warp', 'Light_grey_bg_darker_text', 'Dark_bg_light_text_with_lines']

average_before = [97.262, 96.249, 94.826, 37.803]
average_after = [0, 0.5066, 0, 1.6902]


N = 4
average_before = (97.262, 96.249, 94.826, 37.803)
average_after = (0, 0.5066, 0, 1.6902)

fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35
p1 = ax.bar(ind, average_before, width, bottom=0)
p2 = ax.bar(ind + width, average_after, width, bottom=0)

ax.set_title('Solver Accuracies Before and After Adversarial Noise')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Standard', 'More_Warp', 'More_Contrast', 'Inverted_Lines'))

ax.legend((p1[0], p2[0]), ('Before', 'After'))
ax.autoscale_view()

autolabel(p1)
autolabel(p2)

plt.show()