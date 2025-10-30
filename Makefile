# 定义打包目标：执行后将指定文件/目录压缩为 submission.zip
package:
	zip -r submission.zip \
		code/train.py \
		code/predict.py \
		code/utils.py \
		code/model.py \
		code/requirements.txt \
		model/best_model.pth \
		model/config.json \
		results/

# （可选）添加清理命令，删除临时文件/压缩包
clean:
	rm -f submission.zip