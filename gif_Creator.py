import cv2
import imageio
import os

cap=cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video.avi', fourcc, 20.0, (640, 480))

while True:
    if cv2.waitKey(1)==ord('q'):
        break
    ret,frame=cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out.write(frame)
    cv2.imshow('original',frame)

clip=os.path.abspath('video.avi')

def converter(inp_path,target_format):
    val=''
    s=os.path.splitext(inp_path)[0]
    for i in s[::-1]:
        if i=='/':
            break
        if i not in '.gif':
            val=i+val
    op_path='/home/parmeet/Videos/'+val+target_format
    print(op_path)
    reader=imageio.get_reader(inp_path)
    fps=reader.get_meta_data()['fps']
    writer=imageio.get_writer(op_path,fps=fps)
    print('CONVERTING YOUR FILE TO GIF...PLEASE WAIT!')
    for f in reader:
        writer.append_data(f)
    print('Conversion done!')
    print('Check at location '+op_path+' for your gif')
converter(clip,'.gif')
