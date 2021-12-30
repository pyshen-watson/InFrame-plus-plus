import cv2
import numpy as np
import random
import string

# qHD: 960 * 540
WIDTH = 1920 // 2
HEIGHT = 1080 // 2


CELLNUM = 4         # 4^2 cells form a block
REFERENCENUM = 4    # 4 reference blocks in each row



CELLPIXEL = 10
LOCATORPIXEL = 10
BLOCKSIZE = CELLPIXEL * CELLNUM
LOCATORSIZE = 7 * LOCATORPIXEL


# Correspond to V in HSV, 0-255, the larger the brighter, correspond to delta in the paper
BRIGHTRANGE = np.uint8(20)
#BRIGHTTRANS = [0.4, 0.4, 0.8, 0.8, 1, 1, 0.8, 0.8, 0.4, 0.4]
DATAFRAMENUM = 10



ROWNUM = HEIGHT // BLOCKSIZE
COLUMNNUM = WIDTH // BLOCKSIZE



datablocks = {}
refblock = []
alignblock = []


def Initialization():    
    # Data block
    yshift = np.array([0, 0, 1, 1, 2, 2, 3, 3]) * CELLPIXEL
    xshift = np.array([ [[0, 3, 1, 2, 0, 3, 1, 2], [1, 2, 0, 3, 1, 2, 0, 3]],
                        [[0, 2, 1, 3, 1, 3, 0, 2], [1, 3, 0, 2, 0, 2, 1, 3]],
                        [[0, 2, 1, 3, 0, 2, 1, 3], [1, 3, 0, 2, 1, 3, 0, 2]],
                        [[0, 3, 1, 2, 1, 2, 0, 3], [1, 2, 0, 3, 0, 3, 1, 2]]]) * CELLPIXEL
    for i in range(4):
        black = np.zeros((BLOCKSIZE, BLOCKSIZE), np.uint8)
        white = np.zeros((BLOCKSIZE, BLOCKSIZE), np.uint8)
        # Black
        for j in range(8):
            black[yshift[j] : yshift[j] + CELLPIXEL, xshift[i][0][j] : xshift[i][0][j] + CELLPIXEL] += BRIGHTRANGE
        # White
        for j in range(8):
            white[yshift[j] : yshift[j] + CELLPIXEL, xshift[i][1][j] : xshift[i][1][j] + CELLPIXEL] += BRIGHTRANGE
        datablocks[format(i, "02b")] = [black, white]

    return



def DrawLocator(newframe):
    positions = [[0, 0], [LOCATORSIZE + WIDTH, 0], [0, LOCATORSIZE + HEIGHT], [LOCATORSIZE + WIDTH, LOCATORSIZE + HEIGHT]]
    for x, y in positions:
        for i in range(3):
            shift = i * LOCATORPIXEL
            newframe[y + shift : y + LOCATORSIZE - shift, x + shift : x + LOCATORSIZE - shift] = (0 if i % 2 == 0 else 255)
    return
    



def EncodeData(frame, msg, index, msgid):
    def DrawData(frame, x, y, key, index):
        frame[y : y + BLOCKSIZE, x : x + BLOCKSIZE, 2] = cv2.subtract(\
            frame[y : y + BLOCKSIZE, x : x + BLOCKSIZE, 2],\
            np.uint8(datablocks[key][index % 2]))
            #np.uint8(BRIGHTTRANS[index] * datablocks[key][index % 2]))
        frame[y : y + BLOCKSIZE, x : x + BLOCKSIZE, 2] = cv2.add(\
            frame[y : y + BLOCKSIZE, x : x + BLOCKSIZE, 2],\
            np.uint8(datablocks[key][(index + 1) % 2]))
            #np.uint8(BRIGHTTRANS[index] * datablocks[key][(index + 1) % 2]))
        return

    # Start from left top corner
    x, y = LOCATORSIZE, LOCATORSIZE
    rowbyte = len(msg) // ROWNUM
    for row in range(ROWNUM):
        # Draw reference that contains message id
        msgbits = "{0:08b}".format(msgid % 255)
        for j in range(REFERENCENUM):
            key = msgbits[j * 2 : (j + 1) * 2]
            DrawData(frame, x, y, key, index)
            x += BLOCKSIZE
        # Draw data block
        rowmsg = msg[row * rowbyte : (row + 1) * rowbyte]
        for i in range(len(rowmsg)):
            msgbits = "{0:08b}".format(ord(rowmsg[i]))
            for j in range(len(msgbits) // 2):
                key = msgbits[j * 2 : (j + 1) * 2]
                DrawData(frame, x, y, key, index)
                x += BLOCKSIZE
        y += BLOCKSIZE
        x = LOCATORSIZE
    return




def EditFrame(frame, index, msgid, msg):
    newframe = np.zeros((HEIGHT + LOCATORSIZE * 2, WIDTH + LOCATORSIZE * 2, 3), np.uint8)
    newframe[LOCATORSIZE : LOCATORSIZE + HEIGHT, LOCATORSIZE : LOCATORSIZE + WIDTH] = frame
    DrawLocator(newframe)
    # Change to HSV to adjust luminance
    newframe = cv2.cvtColor(newframe, cv2.COLOR_BGR2HSV)    
    EncodeData(newframe, msg, index, msgid)
    newframe = cv2.cvtColor(newframe, cv2.COLOR_HSV2BGR)
    return newframe


def main():
    Initialization()
    
    # bytecount differ when the encoding method changes
    # Raw encode
    #bytecount = ROWNUM * COLUMNNUM // 4
    bytecount = ROWNUM * (COLUMNNUM - REFERENCENUM) // 4
    # 65 bytes

    msg = []
    #msg_name = "./1560bmsg.txt"
    msg_name = "./longmsg.txt"
    with open(msg_name, "r") as f:
        newmsg = f.read(bytecount)
        while newmsg:
            msg.append(newmsg)
            newmsg = f.read(bytecount)
    

    video_name = "yee"
    video = cv2.VideoWriter(f"./video/{video_name}_{BRIGHTRANGE}_{DATAFRAMENUM}.mp4", -1, 60.0, (WIDTH + 2 * LOCATORSIZE, HEIGHT + 2 * LOCATORSIZE))
    cap = cv2.VideoCapture(f"./video/{video_name}.mp4")
    print(f"BRIGHTRANGE: {BRIGHTRANGE}, DATAFRAMENUM: {DATAFRAMENUM}, bytecount: {bytecount}")
    print(f"read : ./video/{video_name}.mp4")
    print(f"write: ./video/{video_name}_{BRIGHTRANGE}_{DATAFRAMENUM}.mp4")
    
    time = int(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / 60 / 4)
    frame_num = int(time / (DATAFRAMENUM / 60))


    # gray.mp4 for test
    if video_name == "gray":
        frame_num = 24

    print(frame_num)
    msg = msg[:frame_num]


    '''
    # Encode msg into gray frame
    for msgid in range(len(msg)):
        frame = np.full((HEIGHT, WIDTH, 3), 128)
        for i in range(DATAFRAMENUM):
            img = EditFrame(frame, i, msgid, msg[msgid])
            video.write(img)
    '''


    # Circularly encode msg into the whole video frame
    end = False
    msgid = 0
    while not end:
        print(msgid)
        for i in range(DATAFRAMENUM):
            ret, frame = cap.read()
            if not ret:
                print("video end")
                end = True
                break

            img = EditFrame(frame, i, msgid, msg[msgid])
            video.write(img)
            # Show single frame
            #cv2.imshow("newframe", img)
            #cv2.waitKey(0)
        msgid = (msgid + 1) % len(msg)

    video.release()
    cv2.destroyAllWindows()






if __name__ == "__main__":
    main()