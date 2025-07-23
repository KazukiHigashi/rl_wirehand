from tensorboard import program
import time

# ログディレクトリの指定
log_dir = "tensorboard_log"

# TensorBoard プログラムのインスタンスを作成
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])

# 起動
url = tb.launch()

print(f"TensorBoard is running at {url}")

while True:
    time.sleep(1)
