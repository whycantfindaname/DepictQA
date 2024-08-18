import re


def fix_json_missing_commas(input_file, output_file):
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            data = infile.read()

            # 使用正则表达式找到两个 JSON 对象之间缺失的逗号
            # 假设 JSON 对象之间应该用逗号分隔
            # 处理 JSON 数组形式，如 [ { ... } { ... } ]
            fixed_data = re.sub(r"\}\s*\{", "},\n{", data)

            # 加上方括号，使其成为一个有效的 JSON 数组
            fixed_data = f"[{fixed_data}]"

            # 将修复后的 JSON 数据写入新的 JSON 文件
            with open(output_file, "w", encoding="utf-8") as outfile:
                outfile.write(fixed_data)

        print(f"修复完成，标准 JSON 已保存到 {output_file}")

    except Exception as e:
        print(f"发生错误: {e}")


# 使用示例
input_file = "your_input_file.json"  # 替换为你的输入文件路径
output_file = "fixed_json.json"  # 替换为你希望保存的输出文件路径

fix_json_missing_commas(input_file, output_file)
