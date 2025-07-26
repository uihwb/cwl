import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量 - 指定完整路径
load_dotenv(dotenv_path=r"D:\Desktop\paperAnalysis\.env")

# 初始化OpenAI客户端
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY未设置，请检查.env文件")

client = OpenAI(api_key=api_key)

def extract_text_from_pdf(pdf_path):
    """从PDF文件中提取文本"""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def analyze_paper_with_gpt(text, model="gpt-4-0125-preview"):
    """使用GPT API分析论文"""
    prompt = """
    请分析以下学术论文内容，并返回以下方面的总结：
    1. 论文的主要研究目的和贡献
    2. 使用的方法论
    3. 关键发现或结果
    4. 研究的局限性和未来工作建议
    
    请用专业但简洁的语言回答，使用分点格式。
    
    论文内容：
    {text}
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一位专业学术助手，擅长分析和总结科研论文。"},
            {"role": "user", "content": prompt.format(text=text[:120000])}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    # 获取用户输入的PDF文件路径
    pdf_path = input("请输入论文PDF文件的完整路径: ").strip()
    
    # 验证路径是否存在
    if not os.path.exists(pdf_path):
        print("错误: 指定的文件路径不存在!")
        exit()
    
    # 提取文本
    print("正在提取论文文本...")
    try:
        paper_text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        print(f"提取文本失败: {e}")
        exit()
    
    # 分析论文
    print("正在使用GPT分析论文...")
    try:
        analysis = analyze_paper_with_gpt(paper_text)
    except Exception as e:
        print(f"API调用失败: {e}")
        exit()
    
    # 生成结果文件名（原文件名_analysis.txt）
    base_name = os.path.basename(pdf_path)  # 获取文件名（带扩展名）
    file_name_without_ext = os.path.splitext(base_name)[0]  # 去掉扩展名
    result_file_name = f"{file_name_without_ext}_analysis.txt"
    
    # 保存结果
    try:
        with open(result_file_name, "w", encoding="utf-8") as f:
            f.write(analysis)
        print(f"分析完成！结果已保存到 {result_file_name}")
    except Exception as e:
        print(f"保存结果失败: {e}")