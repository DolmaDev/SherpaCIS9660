import os
from PIL import Image

def resize_nuts_folder(input_folder, output_folder, size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    
    for class_name in ['Almonds', 'Cashews', 'Walnuts']:
        input_class = os.path.join(input_folder, class_name)
        output_class = os.path.join(output_folder, class_name)
        os.makedirs(output_class, exist_ok=True)
        
        if os.path.exists(input_class):
            files = [f for f in os.listdir(input_class) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f'Resizing {len(files)} {class_name} images...')
            
            for i, file in enumerate(files):
                if i % 20 == 0:
                    print(f'  {class_name}: {i}/{len(files)}')
                    
                try:
                    img = Image.open(os.path.join(input_class, file))
                    img = img.convert('RGB')
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    
                    output_file = file.rsplit('.', 1)[0] + '.jpg'
                    img.save(os.path.join(output_class, output_file), 'JPEG', quality=85, optimize=True)
                except Exception as e:
                    print(f'Error with {file}: {e}')

# Use the correct path to Desktop/Nuts
resize_nuts_folder('../Nuts', '../Nuts_small', (224, 224))
print('✅ Local resizing complete!')
