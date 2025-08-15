import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_LITE_DISABLE_XNNPACK'] = '1'

import treenutclassifier

print('🚀 Starting training with organized data...')
try:
    model, history = treenutclassifier.train_model('../Nuts_organized', epochs=10)  # Start with fewer epochs
    print('✅ Training completed!')
except Exception as e:
    print(f'❌ Training failed: {e}')
    # Create a demo model instead
    print('Creating demo model...')
    model = treenutclassifier.create_demo_model()
