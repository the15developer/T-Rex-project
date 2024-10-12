import keyboard
import uuid
import time
from PIL import Image
from mss import mss

mon={"top":465, "left":690, "width":250, "height":123}
sct = mss() #bu kutuphane, fotograftan region of interesti kesip frame haline donusturuyor

i = 0

def record_screen(record_id, key):
    global i
    
    i += 1
    print("{}: {}".format(key, i))   
    img = sct.grab(mon)                                #sadece istedigimiz alani aliyor
    im = Image.frombytes("RGB", img.size, img.rgb)
    im.save("./img/{}_{}_{}.png".format(key, record_id, i))  #aldigimiz resmi kaydediyoruz
    
is_exit = False   #veri toplamayi bitirmek istedigimiz anda esc tusuna basiyoruz

def exit():
    global is_exit
    is_exit = True
    
    
keyboard.add_hotkey("esc", exit)

record_id = uuid.uuid4()  #record_id atamasi yapiliyor

while True:
    
    if is_exit: break

    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id, "up")
            time.sleep(0.1)
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id, "down")
            time.sleep(0.1)
        elif keyboard.is_pressed("right"):
            record_screen(record_id, "right")
            time.sleep(0.1)
    except RuntimeError: continue













