# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 

standard_normal = [
            0.125,
            0.409375,
            0.6109375,
            0.6296875,
            0.5921875,
            0.7640625,
            0.6734375,
            0.8890625,
            0.8890625,
            0.840625,
            0.909375,
            0.9359375,
            0.93125,
            0.9359375,
            0.9171875,
            0.8484375,
            0.9625,
            0.8921875,
            0.9265625,
            0.909375,
            0.928125,
            0.878125,
            0.86875,
            0.63125,
            0.9671875,
            0.9125,
            0.8046875,
            0.946875
        ]
standard_adv = [
            0.111979167,
            0.09765625,
            0.153645833,
            0.182291667,
            0.19140625,
            0.1796875,
            0.1796875,
            0.28515625,
            0.325520833,
            0.20703125,
            0.279947917,
            0.2421875,
            0.24609375,
            0.305989583,
            0.274739583,
            0.291666667,
            0.397135417,
            0.27734375,
            0.256510417,
            0.321614583,
            0.302083333,
            0.240885417,
            0.299479167,
            0.264322917,
            0.388020833,
            0.381510417,
            0.251302083,
            0.259114583
        ]
advtrain_normal = [
            0.1046875,
            0.765625,
            0.215625,
            0.8703125,
            0.934375,
            0.9484375,
            0.9484375,
            0.975,
            0.9671875,
            0.9328125,
            0.9609375,
            0.9109375,
            0.95,
            0.9609375,
            0.9453125,
            0.690625,
            0.9859375,
            0.98125,
            0.98125,
            0.940625,
            0.971875,
            0.8296875,
            0.9,
            0.95,
            0.975,
            0.94,
            0.96,
            0.95
        ]
advtrain_adv = [
            0.0625,
            0.4125,
            0.121875,
            0.7,
            0.7953125,
            0.7828125,
            0.8703125,
            0.690625,
            0.7359375,
            0.553125,
            0.8609375,
            0.671875,
            0.5421875,
            0.809375,
            0.65,
            0.78125,
            0.9296875,
            0.85,
            0.9296875,
            0.609375,
            0.8140625,
            0.4359375,
            0.65,
            0.87,
            0.93,
            0.77,
            0.8,
            0.83
        ]

# Data
df=pd.DataFrame(
    {
        'epoch': range(1,29),
        'standard-normal': [i * 100 for i in standard_normal],
        'standard-adv': [i * 100 for i in standard_adv],
        'advtrain-normal': [i * 100 for i in advtrain_normal],
        'advtrain-adv': [i * 100 for i in advtrain_adv]
    }
)
 
plt.title('Normally Trained Solver Accuracy Per Epoch')
plt.plot( 'epoch', 'standard-normal', data=df, marker='', linewidth=2, label='Normal')
plt.plot( 'epoch', 'standard-adv', data=df, marker='', linewidth=2, label='Adversarial')

plt.xlabel('Epoch Number')
plt.xticks(range(1,29))
plt.ylabel('Accuracy (%)')
plt.ylim(0,100)
plt.legend()

plt.show()

plt.clf()

plt.title('Adversarially Trained Solver Accuracy Per Epoch')
plt.plot( 'epoch', 'advtrain-normal', data=df, marker='', linewidth=2, label='Normal')
plt.plot( 'epoch', 'advtrain-adv', data=df, marker='', linewidth=2, label='Adversarial')

plt.xlabel('Epoch Number')
plt.xticks(range(1,29))
plt.ylabel('Accuracy (%)')
plt.ylim(0,100)
plt.legend()

plt.show()