import os
from datetime import datetime


class Printer():
    """
    描述一个将字符串同时输出到终端并保存到文件里的类型
    """
    def __init__(self, file):
        self.file = file
        self.open_or_close = False
        self._check()
        self._open()

    def _check(self):
        """"""
        path, _ = os.path.split(self.file)
        assert os.path.isdir(path)

    def _open(self):
        # 使用追加模式并启用行缓冲，确保写入及时刷新
        self.info = open(self.file, 'w', buffering=1, encoding='utf-8')
        self.open_or_close = True

    def _close(self):
        self.info.close()
        self.open_or_close = False

    def pprint(self, text):
        """
        将字符串输出到屏幕或终端，同时将其作为一行写到文件中
        :param text: 将要输出的字符串
        :return:
        """
        # time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ' ' * 4
        # print(time+text)
        # self.info.write(time+text + '\n')
        print(text)
        self.info.write(text + '\n')
        try:
            self.info.flush()
            os.fsync(self.info.fileno())
        except Exception:
            pass

    def write(self, text):
        self.info.write(text+'\n')
        try:
            self.info.flush()
            os.fsync(self.info.fileno())
        except Exception:
            pass

    def __del__(self):
        try:
            if getattr(self, 'open_or_close', False):
                self._close()
        except Exception:
            pass
