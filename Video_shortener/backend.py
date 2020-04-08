from flask import Flask, render_template, redirect, request, url_for, Response, send_file
import time, pickle, re
import datetime, os
import cv2
import numpy as np
from scipy.spatial import distance
from werkzeug import secure_filename
from pymediainfo import MediaInfo

UPLOAD_FOLDER = 'C:\\Users\\tusha\Desktop\Video_shortener\\upload'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# app.config['MAX_CONTENT_LENGTH'] =  * 1024 * 1024

@app.route("/")
def home():
    return render_template('index.html')


def check_if_video(filename):
    global UPLOAD_FOLDER
    video_file_extensions = (
        '.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.89', '.aaf', '.aec',
        '.aep', '.aepx',
        '.aet', '.aetx', '.ajp', '.ale', '.am', '.amc', '.amv', '.amx', '.anim', '.aqt', '.arcut', '.arf', '.asf',
        '.asx', '.avb',
        '.avc', '.avd', '.avi', '.avp', '.avs', '.avs', '.avv', '.axm', '.bdm', '.bdmv', '.bdt2', '.bdt3', '.bik',
        '.bin', '.bix',
        '.bmk', '.bnp', '.box', '.bs4', '.bsf', '.bvr', '.byu', '.camproj', '.camrec', '.camv', '.ced', '.cel', '.cine',
        '.cip',
        '.clpi', '.cmmp', '.cmmtpl', '.cmproj', '.cmrec', '.cpi', '.cst', '.cvc', '.cx3', '.d2v', '.d3v', '.dat',
        '.dav', '.dce',
        '.dck', '.dcr', '.dcr', '.ddat', '.dif', '.dir', '.divx', '.dlx', '.dmb', '.dmsd', '.dmsd3d', '.dmsm',
        '.dmsm3d', '.dmss',
        '.dmx', '.dnc', '.dpa', '.dpg', '.dream', '.dsy', '.dv', '.dv-avi', '.dv4', '.dvdmedia', '.dvr', '.dvr-ms',
        '.dvx', '.dxr',
        '.dzm', '.dzp', '.dzt', '.edl', '.evo', '.eye', '.ezt', '.f4p', '.f4v', '.fbr', '.fbr', '.fbz', '.fcp',
        '.fcproject',
        '.ffd', '.flc', '.flh', '.fli', '.flv', '.flx', '.gfp', '.gl', '.gom', '.grasp', '.gts', '.gvi', '.gvp',
        '.h264', '.hdmov',
        '.hkm', '.ifo', '.imovieproj', '.imovieproject', '.ircp', '.irf', '.ism', '.ismc', '.ismv', '.iva', '.ivf',
        '.ivr', '.ivs',
        '.izz', '.izzy', '.jss', '.jts', '.jtv', '.k3g', '.kmv', '.ktn', '.lrec', '.lsf', '.lsx', '.m15', '.m1pg',
        '.m1v', '.m21',
        '.m21', '.m2a', '.m2p', '.m2t', '.m2ts', '.m2v', '.m4e', '.m4u', '.m4v', '.m75', '.mani', '.meta', '.mgv',
        '.mj2', '.mjp',
        '.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie',
        '.mp21',
        '.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2', '.mpgindex',
        '.mpl',
        '.mpl', '.mpls', '.mpsub', '.mpv', '.mpv2', '.mqv', '.msdvd', '.mse', '.msh', '.mswmm', '.mts', '.mtv', '.mvb',
        '.mvc',
        '.mvd', '.mve', '.mvex', '.mvp', '.mvp', '.mvy', '.mxf', '.mxv', '.mys', '.ncor', '.nsv', '.nut', '.nuv',
        '.nvc', '.ogm',
        '.ogv', '.ogx', '.osp', '.otrkey', '.pac', '.par', '.pds', '.pgi', '.photoshow', '.piv', '.pjs', '.playlist',
        '.plproj',
        '.pmf', '.pmv', '.pns', '.ppj', '.prel', '.pro', '.prproj', '.prtl', '.psb', '.psh', '.pssd', '.pva', '.pvr',
        '.pxv',
        '.qt', '.qtch', '.qtindex', '.qtl', '.qtm', '.qtz', '.r3d', '.rcd', '.rcproject', '.rdb', '.rec', '.rm', '.rmd',
        '.rmd',
        '.rmp', '.rms', '.rmv', '.rmvb', '.roq', '.rp', '.rsx', '.rts', '.rts', '.rum', '.rv', '.rvid', '.rvl', '.sbk',
        '.sbt',
        '.scc', '.scm', '.scm', '.scn', '.screenflow', '.sec', '.sedprj', '.seq', '.sfd', '.sfvidcap', '.siv', '.smi',
        '.smi',
        '.smil', '.smk', '.sml', '.smv', '.spl', '.sqz', '.srt', '.ssf', '.ssm', '.stl', '.str', '.stx', '.svi', '.swf',
        '.swi',
        '.swt', '.tda3mt', '.tdx', '.thp', '.tivo', '.tix', '.tod', '.tp', '.tp0', '.tpd', '.tpr', '.trp', '.ts',
        '.tsp', '.ttxt',
        '.tvs', '.usf', '.usm', '.vc1', '.vcpf', '.vcr', '.vcv', '.vdo', '.vdr', '.vdx', '.veg', '.vem', '.vep', '.vf',
        '.vft',
        '.vfw', '.vfz', '.vgz', '.vid', '.video', '.viewlet', '.viv', '.vivo', '.vlab', '.vob', '.vp3', '.vp6', '.vp7',
        '.vpj',
        '.vro', '.vs4', '.vse', '.vsp', '.w32', '.wcp', '.webm', '.wlmp', '.wm', '.wmd', '.wmmp', '.wmv', '.wmx',
        '.wot', '.wp3',
        '.wpl', '.wtv', '.wve', '.wvx', '.xej', '.xel', '.xesc', '.xfl', '.xlmv', '.xmv', '.xvid', '.y4m', '.yog',
        '.yuv', '.zeg',
        '.zm1', '.zm2', '.zm3', '.zmv')
    try:
        if filename.endswith((video_file_extensions)):
            return True

        else:
            return False
    except:
        return False

fname=None
@app.route('/shortner', methods=["POST", "GET"])
def shortner():
    global fname
    if request.method == "POST":
        file = request.files["files"]
        filename = secure_filename(file.filename)
        if check_if_video(filename):
            fname=filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            process_and_shorten(filename)
            return render_template('shortner.html', content="Download shortened file by clicking button above")
        else:
            return render_template('shortner.html', content="Issue with the file...try again")
    else:
        return render_template('shortner.html')


@app.route('/download')
def download():
    global fname
    if fname!=None:
        path = UPLOAD_FOLDER + "\\" + 'd' + fname
        fname=None
        return send_file(path, as_attachment=True)
    else:
        return render_template('shortner.html', content="nothing selected...")

@app.route("/about_us")
def about_us():
    return render_template('about_us.html')

def get_distance(curr,prev):
    curr=curr.flatten()
    prev=prev.flatten()
    return distance.euclidean(curr,prev)

def process_and_shorten(filename):
    input_video = UPLOAD_FOLDER + '//' + filename
    cap=cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    print(width,height)
    prev = []
    curr = None
    factor = 10
    thresh = 0
    short_video = []
    cum_dist = 0
    c = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(UPLOAD_FOLDER+'//'+'d'+filename, fourcc, 20.0, (width, height))

    while True:
        try:
            _, img = cap.read()

            w, h, _ = img.shape
            new_img = cv2.resize(img, (h // factor, w // factor), interpolation=cv2.INTER_AREA)

            if len(prev) == 0:
                prev = img
                short_video.append(curr)
                c += 1
            else:
                c += 1
                curr = img
                dist = get_distance(curr, prev)
                cum_dist += dist
                thresh = cum_dist / c
                print(c, len(short_video))
                if dist >= thresh:
                    short_video.append(curr)
                    out.write(curr)
                prev = curr

            if cv2.waitKey(1) == ord('q'):
                break
        except:
            break


if __name__ == "__main__":
    app.run(host='192.168.43.166', debug=True)
