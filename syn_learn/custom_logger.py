import os
import tempfile
import datetime
from typing import Optional, List
from stable_baselines3.common.logger import Logger, make_output_format

def configure_separate_loggers(
    csv_folder: Optional[str] = None,
    tb_folder: Optional[str] = None,
    format_strings: Optional[List[str]] = None
) -> Logger:
    """
    Configure a logger that stores CSV and TensorBoard logs in separate folders.

    :param csv_folder: Folder for CSV logs
    :param tb_folder: Folder for TensorBoard logs
    :param format_strings: Logging formats (default: ['stdout', 'csv', 'tensorboard'])
    :return: Logger object
    """

    # デフォルトフォルダ設定
    if csv_folder is None:
        csv_folder = os.path.join(tempfile.gettempdir(),
                                  datetime.datetime.now().strftime("SB3-csv-%Y-%m-%d-%H-%M-%S-%f"))
    if tb_folder is None:
        tb_folder = os.path.join(tempfile.gettempdir(),
                                 datetime.datetime.now().strftime("SB3-tb-%Y-%m-%d-%H-%M-%S-%f"))

    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(tb_folder, exist_ok=True)

    # デフォルト形式
    if format_strings is None:
        format_strings = ["stdout", "csv", "tensorboard"]

    # 出力フォーマットを作成
    output_formats = []
    for fmt in format_strings:
        if fmt == "csv":
            output_formats.append(make_output_format("csv", csv_folder, ""))
        elif fmt == "tensorboard":
            output_formats.append(make_output_format("tensorboard", tb_folder, ""))
        else:
            # stdoutやlogなど他は共通フォルダ(csv_folder側)に保存
            output_formats.append(make_output_format(fmt, csv_folder, ""))

    logger = Logger(folder=csv_folder, output_formats=output_formats)

    logger.log(f"CSV logs to {csv_folder}")
    logger.log(f"TensorBoard logs to {tb_folder}")

    return logger