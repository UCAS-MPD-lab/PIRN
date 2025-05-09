import os
import re

def get_next_model_filename(model_dir='model', prefix='PIRN', extension='.pth'):
    """
    # 示例用法
    next_model_filename = get_next_model_filename()
    print("Next model file will be saved as:", next_model_filename)
    """
    
    # 获取文件夹中的所有.pth文件
    files = [f for f in os.listdir(model_dir) if f.endswith(extension)]
    
    # 提取文件中的数字部分
    numbers = []
    for file in files:
        match = re.match(rf'{prefix}(\d+){extension}', file)
        if match:
            numbers.append(int(match.group(1)))
    
    # 如果文件夹为空，则返回第一个文件名
    if not numbers:
        return os.path.join(model_dir, f'{prefix}001{extension}')
    
    # 获取最大编号并加1
    next_number = max(numbers) + 1
    return os.path.join(model_dir, f'{prefix}{next_number:03d}{extension}')


