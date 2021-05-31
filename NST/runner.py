import os
import tkinter as tk
import tkinter.font as tkft
from PIL import ImageTk, Image
import tkinter.ttk as ttk
import tkinter.filedialog as fd
import shutil
import threading
import time

exist_img1 = False
exist_img2 = False
exist_img3 = False
activity_button = True

def insert_img(path, idx):
    image = Image.open(path)
    image = image.resize((150, 150))
    image = ImageTk.PhotoImage(image)
    labels[idx].configure(image=image)
    labels[idx].image = image

def load_img(number):
    if not activity_button: return
    
    global exist_img1, exist_img2
    
    path = fd.askopenfilename(initialdir='/',title="select a file",
                              filetypes =[("image files",".jpg"),
                                          ("image files",".png"),])
    
    if number==1 :
        shutil.copy(path, './content_img/image.jpg')
        insert_img('./content_img/image.jpg', 0)
        exist_img1 = True
    elif number==2 :
        shutil.copy(path, './style_img/image.jpg')
        insert_img('./style_img/image.jpg', 1)
        exist_img2 = True

def update(n, epoch, t):
    var1.set('Ongoing...({}/{})\t time: {:.1f} s'.format(n, epoch, time.time()-t))
    lab0.config(fg='black')
    var0.set(n*100//epoch)
    bar.update()

def run(epoch):
    os.system('cmd /c python run.py {}'.format(epoch))

def warning(text):
    var1.set(text)
    lab0.config(fg='red')

def start():
    global activity_button
    
    if not activity_button: return
    
    try:
        epoch = int(ent.get())
    except:
        warning('Write Integer')
        return
    
    if not (0 < epoch <= 100):
        warning('Write Integer (1~100).')
        return

    if not (exist_img1 and exist_img2):
        warning('Press Upload button.')
    else:
        activity_button = False
        labels[2].configure(image=None)
        labels[2].image = None
        with open('./temp.txt', 'w') as f:
            f.write('0')
        
        t1 = threading.Thread(target=run, args=(epoch,))
        t1.start()

        start = time.time()
        before_epoch = 0
        while before_epoch != epoch:
            time.sleep(0.2)
            with open('./temp.txt', 'r') as f:
                current_epoch = int(f.read())
                if before_epoch != current_epoch:
                    before_epoch = current_epoch
                    insert_img('./result_img/image.jpg', 2)
            update(before_epoch, epoch, start)
                    
        activity_button = True
        var1.set('Finish !!!  {:.1f} s'.format(time.time() - start))
        lab0.config(fg='blue')
    return
                   
def save():
    if not activity_button: return
    save_path = fd.asksaveasfilename(initialdir="/", title="Select file",
                                        filetypes =[("image files",".jpg"),])
    if not '.jpg' in save_path: save_path += '.jpg'
    shutil.copy('./result_img/image.jpg', save_path)
    print(save_path)


# init
root = tk.Tk()
root.geometry('650x350')
root.title('Neural Style Transfer')
root.config(bg='#fda')
root.resizable(False, False)

# label
lab = tk.Label(root, text='content image', font=tkft.Font(size=10), bg='#fda')
lab.place(x=35, y=10, width=100, height=30)

lab = tk.Label(root, text='style image', font=tkft.Font(size=10), bg='#fda')
lab.place(x=275, y=10, width=100, height=30)

lab = tk.Label(root, text='result image', font=tkft.Font(size=10), bg='#fda')
lab.place(x=515, y=10, width=100, height=30)

lab = tk.Label(root, text='+', anchor='w', font=tkft.Font(size=20), bg='#fda')
lab.place(x=190, y=110, width=20, height=20)

lab = tk.Label(root, text='=', anchor='w', font=tkft.Font(size=20), bg='#fda')
lab.place(x=430, y=110, width=20, height=20)

# image places
labels = [tk.Label(root, image=None) for _ in range(3)]
labels[0].place(x=10, y=50, width=150, height=150)
labels[1].place(x=250, y=50, width=150, height=150)
labels[2].place(x=490, y=50, width=150, height=150)
if 'image.jpg' in os.listdir('./content_img'):
    insert_img('./content_img/image.jpg', 0)
    exist_img1 = True
if 'image.jpg' in os.listdir('./style_img'):
    insert_img('./style_img/image.jpg', 1)
    exist_img2 = True
if 'image.jpg' in os.listdir('./result_img'):
    insert_img('./result_img/image.jpg', 2)
    exist_img3 = True

# buttons
btn = tk.Button(root, text='Upload', font=tkft.Font(size=10), command=lambda:load_img(1))
btn.place(x=35, y=205, width=100, height=30)

btn = tk.Button(root, text='Upload', font=tkft.Font(size=10), command=lambda:load_img(2))
btn.place(x=275, y=205, width=100, height=30)

btn = tk.Button(root, text='Start', font=tkft.Font(size=10), command=start)
btn.place(x=515, y=205, width=100, height=30)

btn = tk.Button(root, text='Save', font=tkft.Font(size=10), command=save)
btn.place(x=515, y=240, width=100, height=30)

# label
lab = tk.Label(root, text='times (1~100)', font=tkft.Font(size=10), bg='#fda')
lab.place(x=515, y=270, width=100, height=30)

# epoch
ent = tk.Entry(root)
ent.place(x=515, y=300, width=100, height=30)
ent.insert(0, '1')

# progressbar
var0 = tk.DoubleVar()
bar = ttk.Progressbar(root, maximum=100, mode='determinate', variable=var0)
bar.place(x=35, y=300, width=400, height=30)

# state
var1 = tk.StringVar(value='state')
lab0 = tk.Label(root, textvariable=var1, anchor='w', font=tkft.Font(size=10), bg='#fda')
lab0.place(x=35, y=270, width=200, height=30)


root.mainloop()



