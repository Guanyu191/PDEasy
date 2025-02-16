import os

def count_lines_in_file(file_path):
    """
    统计单个文件的代码行数
    :param file_path: 文件的路径
    :return: 文件的代码行数
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # 过滤掉空行和仅包含注释的行
            code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            return len(code_lines)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return 0

def count_py_lines_in_current_dir():
    """
    统计当前文件夹内所有 .py 文件的代码行数，并打印文件名称
    :return: 所有 .py 文件的代码总行数
    """
    total_lines = 0
    current_dir = os.getcwd()
    print("当前文件夹内的 .py 文件有：")
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(file_path)
                lines = count_lines_in_file(file_path)
                total_lines += lines
    return total_lines

if __name__ == "__main__":
    total_lines = count_py_lines_in_current_dir()
    print(f"\n当前文件夹内所有 .py 文件的代码总行数为: {total_lines}")