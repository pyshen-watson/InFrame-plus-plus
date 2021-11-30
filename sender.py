import cv2
import numpy as np
import random
import string

# qHD: 960 * 540
WIDTH = 1920 // 2
HEIGHT = 1080 // 2


CELLNUM = 4         # 4^2 pixels form a cell
BLOCKNUM = 4        # 4^2 cells form a block
SEGMENTNUM = 10     # 10 blocks form a segment, 8 blocks = 16 bits for data, 2 blocks for parity, 4.1.4.1
REFERENCENUM = 6    # 6 reference blocks in each row

DATAFRAMENUM = 10
TRANSFRAMENUM = 4   # 4 frames for transition

CELLPIXEL = 3
ALIGNPIXEL = 2 * CELLPIXEL
LOCATORPIXEL = (ALIGNPIXEL * 4) // 3
BLOCKSIZE = CELLPIXEL * CELLNUM
ALIGNSIZE = ALIGNPIXEL * CELLNUM
LOCATORSIZE = 7 * LOCATORPIXEL

GUARDSIZE = 2 * BLOCKSIZE
REFERENCESIZE = BLOCKSIZE

# Correspond to V in HSV, 0-255, the larger the brighter, correspond to delta in the paper
BRIGHTRANGE = np.uint8(20)
#BRIGHTTRANS = [0.3, 0.8, 1, 1, 1, 1, 1, 1, 0.8, 0.3]
#BRIGHTTRANS = [0.2, 0.4, 0.6, 0.8, 1, 1, 0.8, 0.6, 0.4, 0.2]
BRIGHTTRANS = [0.4, 0.4, 1, 1, 1, 1, 1, 1, 0.4, 0.4]


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


    # Reference block
    yshift = np.array([0, 0, 1, 1, 2, 2, 3, 3]) * CELLPIXEL
    xshift = np.array([[0, 3, 0, 3, 1, 2, 1, 2], [1, 2, 1, 2, 0, 3, 0, 3]]) * CELLPIXEL
    black = np.zeros((BLOCKSIZE, BLOCKSIZE), np.uint8)
    white = np.zeros((BLOCKSIZE, BLOCKSIZE), np.uint8)
    # Black
    for i in range(8):
        black[yshift[i] : yshift[i] + CELLPIXEL, xshift[0][i] : xshift[0][i] + CELLPIXEL] += BRIGHTRANGE
    # White    
    for i in range(8):
        white[yshift[i] : yshift[i] + CELLPIXEL, xshift[1][i] : xshift[1][i] + CELLPIXEL] += BRIGHTRANGE
    refblock.append(black)
    refblock.append(white)


    # Alignment block
    yshift = np.array([0, 0, 1, 1, 2, 2, 3, 3]) * ALIGNPIXEL
    xshift = np.array([[0, 2, 1, 3, 0, 2, 1, 3], [1, 3, 0, 2, 1, 3, 0, 2]]) * ALIGNPIXEL
    black = np.zeros((ALIGNSIZE, ALIGNSIZE), np.uint8)
    white = np.zeros((ALIGNSIZE, ALIGNSIZE), np.uint8)
    # Black
    for i in range(8):
        black[yshift[i] : yshift[i] + ALIGNPIXEL, xshift[0][i] : xshift[0][i] + ALIGNPIXEL] += BRIGHTRANGE
    # White
    for i in range(8):
        white[yshift[i] : yshift[i] + ALIGNPIXEL, xshift[1][i] : xshift[1][i] + ALIGNPIXEL] += BRIGHTRANGE
    alignblock.append(black)
    alignblock.append(white)

    return



def DrawLocator(newframe):
    positions = [[0, 0], [LOCATORSIZE + WIDTH, 0], [0, LOCATORSIZE + HEIGHT], [LOCATORSIZE + WIDTH, LOCATORSIZE + HEIGHT]]
    for x, y in positions:
        for i in range(3):
            shift = i * LOCATORPIXEL
            newframe[y + shift : y + LOCATORSIZE - shift, x + shift : x + LOCATORSIZE - shift] = (0 if i % 2 == 0 else 255)
    return
    
def DrawAllAlign(frame, index):
    def DrawAlign(x, y, index):
        frame[y : y + ALIGNSIZE, x : x + ALIGNSIZE, 2] = cv2.subtract(\
            frame[y : y + ALIGNSIZE, x : x + ALIGNSIZE, 2],\
            alignblock[index % 2])
        frame[y : y + ALIGNSIZE, x : x + ALIGNSIZE, 2] = cv2.add(\
            frame[y : y + ALIGNSIZE, x : x + ALIGNSIZE, 2],\
            alignblock[(index + 1) % 2])
        return
    # Horizontal
    x = LOCATORSIZE + ALIGNPIXEL * CELLNUM
    top, bottom = 2 * LOCATORPIXEL, LOCATORSIZE + HEIGHT + 2 * LOCATORPIXEL
    while x < LOCATORSIZE + WIDTH:
        DrawAlign(x, top, index)
        DrawAlign(x, bottom, index)
        x += 2 * ALIGNPIXEL * CELLNUM
    # Vertical
    y = LOCATORSIZE + ALIGNPIXEL * CELLNUM
    left, right = 2 * LOCATORPIXEL, LOCATORSIZE + WIDTH + 2 * LOCATORPIXEL
    while y < LOCATORSIZE + HEIGHT:
        DrawAlign(left, y, index)
        DrawAlign(right, y, index)
        y += 2 * ALIGNPIXEL * CELLNUM
    return




def EncodeData(frame, msg, index):
    def DrawData(frame, x, y, key, index):
        frame[y : y + BLOCKSIZE, x : x + BLOCKSIZE, 2] = cv2.subtract(\
            frame[y : y + BLOCKSIZE, x : x + BLOCKSIZE, 2],\
            np.uint8(BRIGHTTRANS[index] * datablocks[key][index % 2]))
        frame[y : y + BLOCKSIZE, x : x + BLOCKSIZE, 2] = cv2.add(\
            frame[y : y + BLOCKSIZE, x : x + BLOCKSIZE, 2],\
            np.uint8(BRIGHTTRANS[index] * datablocks[key][(index + 1) % 2]))
        return
    def DrawReference(frame, x, y, index):
        frame[y : y + BLOCKSIZE, x : x + BLOCKSIZE, 2] = cv2.subtract(\
            frame[y : y + BLOCKSIZE, x : x + BLOCKSIZE, 2],\
            np.uint8(BRIGHTTRANS[index] * refblock[index % 2]))
        frame[y : y + BLOCKSIZE, x : x + BLOCKSIZE, 2] = cv2.add(\
            frame[y : y + BLOCKSIZE, x : x + BLOCKSIZE, 2],\
            np.uint8(BRIGHTTRANS[index] * refblock[(index + 1) % 2]))
        return
    # Start from left top corner
    x, y = LOCATORSIZE + GUARDSIZE, LOCATORSIZE + GUARDSIZE
    for i in range(len(msg)):
        msgbits = "{0:08b}".format(ord(msg[i]))
        for j in range(len(msgbits) // 2):
            key = msgbits[j * 2 : (j + 1) * 2]
            DrawData(frame, x, y, key, index)
            x += BLOCKSIZE

        # Parity for 4 data blocks in front
        parity = "{0:02b}".format(msgbits.count("1") % 4)
        DrawData(frame, x, y, parity, index)
        x += BLOCKSIZE

        # Every 10 blocks, add a reference if not meet right boundary
        if i % 2 == 1:
            if x == LOCATORSIZE + WIDTH - GUARDSIZE:
                y += BLOCKSIZE
                x = LOCATORSIZE + GUARDSIZE
            else:
                # Use 00 datablock for reference
                DrawData(frame, x, y, "00", index)
                #DrawReference(frame, x, y, index)
                x += BLOCKSIZE
    return



def EditFrame(frame, msg, nextmsg, index):
    newframe = np.zeros((HEIGHT + LOCATORSIZE * 2, WIDTH + LOCATORSIZE * 2, 3), np.uint8)
    newframe[LOCATORSIZE : LOCATORSIZE + HEIGHT, LOCATORSIZE : LOCATORSIZE + WIDTH] = frame

    DrawLocator(newframe)
    # Change to HSV to adjust luminance
    newframe = cv2.cvtColor(newframe, cv2.COLOR_BGR2HSV)    
    DrawAllAlign(newframe, index)
    EncodeData(newframe, msg, index)
    newframe = cv2.cvtColor(newframe, cv2.COLOR_HSV2BGR)
    return newframe

'''
TODO:
ignore alignment
receiver: raw video -> recorded video by camera
'''



def main():

    Initialization()
    fp = open("./msg.txt", "r")
    

    # bytecount differ when the encoding method changes
    # Raw encode
    #bytecount = (HEIGHT // BLOCKSIZE) * (WIDTH // BLOCKSIZE) // 4
    # Add visual guard
    #bytecount = ((HEIGHT - 2 * GUARDSIZE) // BLOCKSIZE) * ((WIDTH - 2 * GUARDSIZE) // BLOCKSIZE) // 4
    # Add reference blocks
    # for 960 * 540, it becomes 76 * 41 blocks for data + ref, 
    #bytecount = ((HEIGHT - 2 * GUARDSIZE) // BLOCKSIZE) * ((WIDTH - 2 * GUARDSIZE - REFERENCENUM * REFERENCESIZE) // BLOCKSIZE) // 4
    # Add parity
    bytecount = ((HEIGHT - 2 * GUARDSIZE) // BLOCKSIZE) * \
                ((WIDTH - 2 * GUARDSIZE - REFERENCENUM * REFERENCESIZE - 2 * (REFERENCENUM + 1) * BLOCKSIZE) // BLOCKSIZE) // 4

    print(bytecount)

    video = cv2.VideoWriter('./encode60_trans_20.mp4', -1, 60.0, (WIDTH + 2 * LOCATORSIZE, HEIGHT + 2 * LOCATORSIZE))
    cap = cv2.VideoCapture("./yee960_60.mp4")

    end = False
    allchr = [chr(i) for i in range(256)]

    msg = fp.read(bytecount)
    msg = [random.choice(allchr) for _ in range(bytecount)]
    while not end:
        nextmsg = fp.read(bytecount)
        nextmsg = [random.choice(allchr) for _ in range(bytecount)]

        for i in range(DATAFRAMENUM):
            # All gray image to test
            #frame = np.full((HEIGHT, WIDTH, 3), 196, np.uint8)

            ret, frame = cap.read()
            if not ret:
                print("video end")
                end = True
                break

            img = EditFrame(frame, msg, nextmsg, i)
            video.write(img)

            # Show single frame
            #cv2.imshow("newframe", img)
            #cv2.waitKey(0)


        msg = nextmsg
        end |= (len(msg) == 0)

    video.release()

    cv2.destroyAllWindows()






if __name__ == "__main__":
    main()