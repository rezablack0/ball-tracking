# paket import yang di butuhkan
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())


greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

# jika jalur video tidak disediakan, ambil referensinya
# ke webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# jika tidak, ambil referensi ke file video
else:
	vs = cv2.VideoCapture(args["video"])
# buat pewaktu video
time.sleep(2.0)

#looping
while True:
	# mengunakan frame
	frame = vs.read()
	# menghandle frame dari videostream atau videocapture
	frame = frame[1] if args.get("video", False) else frame
	# jika kita mengambil /merekam video tidak ada frame,
	# maka kita telah mencapai akhir video
	if frame is None:
		break
	# resize frame, buat blur, dan mengconvert ke HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# buat topeng untuk warna "hijau", lalu lakukan
	# serangkaian pelebaran dan erosi untuk menghilangkan yang kecil
	# gumpalan tersisa di topeng
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)


# temukan kontur di topeng dan inisialisasi arus
	# (x, y) pusat bola
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	# hanya lanjutkan jika setidaknya satu kontur ditemukan
	if len(cnts) > 0:
		# temukan kontur terbesar di topeng, lalu gunakan
		# itu untuk menghitung lingkaran penutup minimum dan
		# pusat
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# hanya lanjutkan jika radius memenuhi ukuran minimum
		if radius > 10:
			# menggambar lingkaran dan centroid pada FRAME,
			# kemudian perbarui daftar poin yang dilacak
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
	# perbarui antrian poin
	pts.appendleft(center)


#loop di atas set poin yang dilacak
	for i in range(1, len(pts)):
		# jika salah satu dari titik yang dilacak adalah Tidak Ada, abaikan
		# mereka
		if pts[i - 1] is None or pts[i] is None:
			continue
		# jika tidak, hitung ketebalan garis dan
		#gambar garis penghubung
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	# menunjukan frame ke layar
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is ditekan, maka loop akan stop
	if key == ord("q"):
		break
#jika tidak menggunakan file video.maka video akan di stop
if not args.get("video", False):
	vs.stop()
# jika tidak ,maka camera di lepas
else:
	vs.release()
# menutup semua windows
cv2.destroyAllWindows()


