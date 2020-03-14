from core import Emanalisis
import platform
import datetime
emails = ['iasizykh@miem.hse.ru', 'internet.prisoner59@gmail.com']

cam504 = '172.18.191.26'

a = Emanalisis(output_mode=0, input_mode=1, channel=cam504, record_video=True, on_gpu=False, display=True,
               send_to_nvr=False, email_to_share=emails)
while True:
    now = datetime.datetime.now()
    if 9 <= int(now.strftime("%H")) <= 18:
        # if now.strftime("%m") == "00" or now.strftime("%m") == "30":
        if platform.system() == 'Windows':
            filename = now.strftime("%Y-%m-%d_%H-%M")
        else:
            filename = now.strftime("%Y-%m-%d_%H:%M")

        a.run(filename=filename, fps_factor=25, stop_time=1700)
