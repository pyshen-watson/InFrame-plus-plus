import cv2
import numpy as np
import collections
import string


# qHD: 960 * 540
WIDTH = 1920 // 2
HEIGHT = 1080 // 2


CELLNUM = 4         # 4^2 cells form a block
REFERENCENUM = 4    # 4 reference blocks in the beginning of each row


CELLPIXEL = 10
LOCATORPIXEL = 10
BLOCKSIZE = CELLPIXEL * CELLNUM
LOCATORSIZE = 7 * LOCATORPIXEL


# Correspond to V in HSV, 0-255, the larger the brighter, correspond to delta in the paper
BRIGHTRANGE = np.uint8(5)
DATAFRAMENUM = 20

# Receiver

ROWNUM = HEIGHT // BLOCKSIZE
COLUMNNUM = WIDTH // BLOCKSIZE

SENDERFPS = 60
RECEIVERFPS = 30





# Top, Bottom, Left boundary, Right boundary
def FindVideoPosition(frame):
    def LocatorTemplate():
        template = np.zeros((LOCATORSIZE, LOCATORSIZE), np.uint8)
        template[LOCATORPIXEL : LOCATORSIZE - LOCATORPIXEL, LOCATORPIXEL : LOCATORSIZE - LOCATORPIXEL] = 255
        template[2 * LOCATORPIXEL : LOCATORSIZE - 2 * LOCATORPIXEL, 2 * LOCATORPIXEL : LOCATORSIZE - 2 * LOCATORPIXEL] = 0
        return template

    template = LocatorTemplate()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = np.shape(gray_frame)

    top, bot, left, right = [], [], [], []
    scalings = []
    for scaling in np.linspace(0.5, 1.5, 20):
        scaled_size = int(scaling * LOCATORSIZE)
        resized = cv2.resize(template, (scaled_size, scaled_size))
        result = cv2.matchTemplate(gray_frame, resized, cv2.TM_CCOEFF_NORMED)
        y_coor, x_coor = np.where(result >= 0.9)
        if len(y_coor) > 0:
            scalings.append(scaling)
            for y in y_coor:
                if y < h / 2:
                    top.append(y)
                else:
                    bot.append(y)
            for x in x_coor:
                if x < w / 2:
                    left.append(x)
                else:
                    right.append(x)

    scaling = np.mean(scalings)
    video_position = [  int(np.mean(top) + scaling * LOCATORSIZE), int(np.mean(bot)),\
                        int(np.mean(left) + scaling * LOCATORSIZE), int(np.mean(right)) ]
    return scaling, video_position


# Datablocks are different for sender and receiver
def DataBlockTemplate(scaling):
    datablocks = {}
    scaled_blocksize = int(scaling * BLOCKSIZE)

    # 1 for white, -1 for black
    templates = []
    templates.append([ [-1, 1, 1, -1], [1, -1, -1, 1], [-1, 1, 1, -1], [1, -1, -1, 1] ])
    templates.append([ [-1, 1, -1, 1], [1, -1, 1, -1], [1, -1, 1, -1], [-1, 1, -1, 1] ])
    templates.append([ [-1, 1, -1, 1], [1, -1, 1, -1], [-1, 1, -1, 1], [1, -1, 1, -1] ])
    templates.append([ [-1, 1, 1, -1], [1, -1, -1, 1], [1, -1, -1, 1], [-1, 1, 1, -1] ])
    for i in range(4):
        block = np.zeros((scaled_blocksize, scaled_blocksize), np.int16)
        for j in range(scaled_blocksize):
            for k in range(scaled_blocksize):
                jj = int(j / scaled_blocksize * 4)
                kk = int(k / scaled_blocksize * 4)
                block[j][k] = templates[i][jj][kk] * BRIGHTRANGE
        datablocks[format(i, "02b")] = block
    return datablocks


def EmptyRecord():
    record = [[("00", 0) for _ in range(COLUMNNUM - REFERENCENUM)] for _ in range(ROWNUM)]
    return record


def DecodeFrame(frame, datablocks, scaling):
    def DecodeBlock(x, y):
        results = []
        offset = 2
        for x_pos in range(x - offset, x + 1 + offset):
            for y_pos in range(y - offset, y + 1 + offset):
                if x_pos < 0 or x_pos + scaled_blocksize >= w or y_pos < 0 or y_pos + scaled_blocksize >= h:
                    continue
                block = cv2.cvtColor(frame[y_pos : y_pos + scaled_blocksize, x_pos : x_pos + scaled_blocksize], cv2.COLOR_BGR2HSV)
                block = block[:, :, 2].astype(np.int16)
                mean_bright = int(np.mean(block))
                block -= mean_bright
                for key, val in datablocks.items():
                    res = abs(np.sum(np.multiply(block, val))) / max_correlation
                    results.append([x_pos, y_pos, key, res])
        results.sort(key = lambda k: k[3], reverse = True)

        # draw black dots to check where
        #xx, yy = results[0][0], results[0][1]
        #frame[yy:yy+2, xx:xx+2] = 0
        return results[0][2], results[0][3]

    scaled_blocksize = int(scaling * BLOCKSIZE)
    max_correlation = (BRIGHTRANGE ** 2) * (scaled_blocksize ** 2)

    msgids = []
    record = EmptyRecord()

    h, w, _ = np.shape(frame)
    x, y = 0, 0
    xgap, ygap = w / COLUMNNUM, (h - scaled_blocksize / 2) / ROWNUM
    for row in range(ROWNUM):
        # Get msgid from first 4 reference block
        bits = ""
        for i in range(REFERENCENUM):
            bits += DecodeBlock(int(x), int(y))[0]
            x += xgap
        msgids.append(int(bits, 2))
        # Get data block in a row
        rowmsg = ""
        for col in range(COLUMNNUM - REFERENCENUM):
            record[row][col] = DecodeBlock(int(x), int(y))
            x += xgap
            
        y += ygap
        x = 0
        
    refcount = collections.Counter(msgids).most_common(1)[0]
    msgid = refcount[0] if refcount[1] > ROWNUM // 2 else None
    
    #cv2.imshow("f", frame)
    #cv2.waitKey(0)

    return msgid, record


def RecordCompare(record, msgrecord):
    for row in range(ROWNUM):
        for col in range(COLUMNNUM - REFERENCENUM):
            if msgrecord[row][col][1] > record[row][col][1]:
                record[row][col] = msgrecord[row][col]
    return record



def Translate(record):
    bits = ""
    for row in range(ROWNUM):
        for col in range(COLUMNNUM - REFERENCENUM):
            bits += record[row][col][0]
    msg = ""
    for i in range(len(bits) // 8):
        c = chr(int(bits[i * 8 : (i + 1) * 8], 2))
        if c not in string.ascii_letters:
            c = "\\"
        msg += c
    return msg


def ErrorRate(received_message, bytecount, frame_num):
    #msg_name = "./1560bmsg.txt"
    msg_name = "./longmsg.txt"
    with open(msg_name, "r") as f:
        total_err, corrupt_frame = 0, 0
        for i in range(frame_num):
            err = 0
            orimsg = f.read(bytecount)
            print(f"{i:<2}-th frame ===========")
            print(f"truth: {orimsg}")
            print(f"recvd: {received_message[i]}")
            if not received_message[i] or len(orimsg) != len(received_message[i]):
                print(f"length error")
                corrupt_frame += 1
                continue
            for j in range(len(orimsg)):
                if orimsg[j] != received_message[i][j]:
                    err += 1
                    total_err += 1
            print(f"err: {err}, rate = {err / bytecount}")

    # Some statistics
    if corrupt_frame != frame_num:
        excluded_rate = total_err / (bytecount * (frame_num - corrupt_frame))
    else:
        excluded_rate = 1
    overall_rate = (total_err + bytecount * corrupt_frame) / (bytecount * frame_num)
    return [corrupt_frame, excluded_rate, overall_rate]


'''
TODO:
different video background? (black may cause higher error rate?)
experiment parameters: BRIGHTRANGE(5, 10, 20), DATAFRAMENUM(4?, 6, 10, 20)


plot error rate vs. loop_time?
plot overall error rate under different parameters
using graph or table?

larger BRIGHTRANGE will cause flicker more frequent
transition frame is removed for higher accuracy
when decoding, frames are perodically break, because of camera? or screen?
thus receive the video for multiple times to combine results to get correct message


'''


def main():
    # 65 bytes
    bytecount = ROWNUM * (COLUMNNUM - REFERENCENUM) // 4

    # camera frame
    video_name = "gray"
    raw_video_name = f"{video_name}_{BRIGHTRANGE}_{DATAFRAMENUM}"
    #cap = cv2.VideoCapture(f"./video/{raw_video_name}.mp4")
    cap = cv2.VideoCapture(f"./video/camera_{raw_video_name}.mp4")
    print(f"BRIGHTRANGE: {BRIGHTRANGE}, DATAFRAMENUM: {DATAFRAMENUM}, bytecount: {bytecount}")
    print(f"read : camera_{raw_video_name}.mp4")

    # other video
    time = int(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / RECEIVERFPS / 4)
    frame_num = int(time / (DATAFRAMENUM / SENDERFPS))

    # gray.mp4 for test
    if video_name == "gray":
        frame_num = 24
    
    print(frame_num)

    ret, original_frame = cap.read()
    scaling, video_position = FindVideoPosition(original_frame)
    datablocks = DataBlockTemplate(scaling)

    recordid, record = 0, EmptyRecord()
    best_records = collections.defaultdict(EmptyRecord)

    loop_time = 0
    count = 0
    count_floor = np.round(DATAFRAMENUM * RECEIVERFPS * 0.4) // SENDERFPS

    # each entry contains [number of corrupted frames, overall error rate]
    error_log = []
    while ret:
        frame = original_frame[video_position[0] : video_position[1], video_position[2] : video_position[3]]
        msgid, msgrecord = DecodeFrame(frame, datablocks, scaling)
        if msgid == None:
            print(f"Unable to decode the frame, skip...")
            ret, original_frame = cap.read()
            continue
        if msgid != recordid:
            # Pick the best record in all loops of video play
            # If the number of the record is too few (due to camera? or screen?), then discard this record
            if count >= count_floor:
                best_records[recordid] = RecordCompare(best_records[recordid], record)
            if msgid < recordid:
                received_message = collections.defaultdict(dict)
                print(f"{loop_time}-th loop -------------------------")
                for recordid, record in best_records.items():
                    received_message[recordid] = Translate(best_records[recordid])
                er = ErrorRate(received_message, bytecount, frame_num)
                error_log.append(er)
                print(f"{loop_time}-th loop -------------------------")
                loop_time += 1

            record = msgrecord
            recordid = msgid
            count = 1
        else:
            # Pick the best record in continuous identical frames
            record = RecordCompare(record, msgrecord)
            count += 1
        print(msgid, count)
        
        ret, original_frame = cap.read()


    # Translate record to received message
    received_message = collections.defaultdict(dict)
    print("-------------------------")
    for recordid, record in best_records.items():
        received_message[recordid] = Translate(best_records[recordid])
        print(f"{recordid:<2}: {received_message[recordid]}")
    print("-------------------------")


    # error rate
    ErrorRate(received_message, bytecount, frame_num)
    print(f"BRIGHTRANGE: {BRIGHTRANGE}, DATAFRAMENUM: {DATAFRAMENUM}, bytecount: {bytecount}, frame_num: {frame_num}")
    print(f"[number of corrupted frame, excluded error rate, overall error rate]")
    for er in error_log:
        print(er)
    return




if __name__ == '__main__':
    main()