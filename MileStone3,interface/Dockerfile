# 1. 使用官方 Python 轻量级镜像
FROM python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 复制依赖文件到容器中
COPY requirements.txt .

# 4. 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制当前目录下的所有文件到容器中
COPY . .

# 6. 暴露 Streamlit 的默认端口
EXPOSE 8501

# 7. 运行 Streamlit 应用
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]