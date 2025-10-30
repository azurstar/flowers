# Makefile for creating the submission.zip file (The Strictest Version)

# 定义目标文件名
ZIP_FILE = submission.zip

# 定义需要包含的目录
DIRS = code model results

# 定义 code 目录下需要包含的文件
CODE_FILES = model.py predict.py requirements.txt train.py utils.py

# 定义 model 目录下需要包含的文件
# 任何文件缺失都将导致 make 失败
MODEL_FILES = best_model.pth config.json

# 默认目标：创建 zip 文件
all: $(ZIP_FILE)

# 清理目标：只删除生成的 zip 文件
clean:
	@echo "Cleaning up only the zip file..."
	@rm -f $(ZIP_FILE)

# 创建 zip 文件
$(ZIP_FILE): $(DIRS)
	@echo "Creating required directories and copying files..."
	
	# 复制 code 文件。如果任何文件不存在，cp 命令将失败，make 停止。
	@cp $(CODE_FILES) code/
	
	# 复制 model 文件。如果任何文件不存在，cp 命令将失败，make 停止。
	@cp $(MODEL_FILES) model/
	
	@echo "Zipping the submission package..."
	# 使用 -r 递归地将目录添加到 zip 文件中
	@zip -r -q $(ZIP_FILE) $(DIRS)
	@echo "Successfully created $(ZIP_FILE)"

# 确保目录存在
$(DIRS):
	@mkdir -p $@

.PHONY: all clean
