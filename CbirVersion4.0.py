from tkinter import *
from tkinter import ttk
from tkinter import filedialog,messagebox
from PIL import ImageTk, Image
import glob, os, math
import cv2
import numpy as np
# from pandas import DataFrame
import matplotlib 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.figure import Figure
from skimage import feature  
from csv import reader,writer


class CBIR(Frame):

    def __init__(self, root,pix):

        # variables
        self.pix = pix
        self.imageList = pix.get_imageList()
        self.photoList = pix.get_photoList()
        self.colorBins = pix.get_colorCode()
        self.colorBinsHsv = pix.get_colorCodeHsv()
        self.textureBins = pix.get_texture()
        self.textureBinsLbp = pix.get_texturelbp()
        self.shapeBins = pix.get_shape()
        self.shapeBinsCanny = pix.get_shapeCanny()
        self.allBins = []
        self.xmax = pix.get_xmax()+20
        self.ymax = pix.get_ymax()+10
        self.bgc = '#4e7a8a'
        self.btc = '#161616'
        self.abtc= '#c6ffeb'
        self.fgc = '#ffffff'
        self.bth = 5
        self.currentPage = 0
        self.currentPhotoList = pix.get_photoList()
        self.currentImageList = pix.get_imageList()
        self.totalPages = self.get_totalPages()
        self.iteration = 0
        self.weights = []
        self.selected = 0
        self.precision = 0
        self.recall = 0
        self.checkbox_var = [IntVar(),IntVar(),IntVar(),IntVar(),IntVar(),IntVar()]
        self.pos = 0


        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True)

        # create frames
        self.frameMain = ttk.Frame(self.notebook)
        self.frameStat = ttk.Frame(self.notebook)
        self.frameImage = ttk.Frame(self.notebook)

        self.frameMain.pack(fill='both', expand=True)
        self.frameStat.pack(fill='both', expand=True)
        self.frameImage.pack(fill='both', expand=True)

        # add frames to notebook
        self.notebook.bind("<<NotebookTabChanged>>",self.on_tab_selected)
        self.notebook.add(self.frameMain, text='General')
        self.notebook.add(self.frameStat, text='Histogram')
        self.notebook.add(self.frameImage, text='Preview')

        # main frame
        self.mainframe = Frame(self.frameMain,bg=self.bgc)
        self.mainframe.pack()

        # section frames
        self.topFrame = Frame(self.mainframe,bg=self.bgc)
        self.topFrame.pack(side=LEFT)
        self.bottomFrame = Frame(self.mainframe,bg=self.bgc)
        self.bottomFrame.pack(side = LEFT)

        # selected image
        self.selectedView = Label(self.topFrame,width=450, height=pix.get_y(),bg=self.bgc)
        self.selectedView.pack(side=TOP)
        self.update_preview(self.imageList[0].filename)
   
        # control panel
        self.controlPanel = Frame(self.topFrame,bg=self.bgc)
        self.controlPanel.pack()
        self.b_bro = Button(self.controlPanel,text="Browse",
                           width=40,pady=self.bth,border=0,bg=self.btc,
                           fg=self.fgc,activebackground=self.abtc,
                           command = lambda:self.file_dialog())
        self.b_bro.config(command=self.file_dialog)
        self.b_bro.pack()
        
        self.l0 = Label(self.controlPanel,bg=self.bgc,text=' ')
        self.l0.pack()

        self.frame0 = LabelFrame(self.controlPanel,text = "Color Feature",bg = self.bgc,bd =3)
        self.frame0.pack(fill = BOTH,expand = 1)
        self.color_chechbox1 = Checkbutton(self.frame0,text= "RGB",variable = self.checkbox_var[0],onvalue = 1, offvalue = 0,bg=self.bgc,padx=40)
        self.color_chechbox1.pack(side = LEFT)
        self.color_chechbox2 = Checkbutton(self.frame0,text= "HSV",variable = self.checkbox_var[1],onvalue = 1, offvalue = 0,bg=self.bgc,padx=55)
        self.color_chechbox2.pack(side = RIGHT)

        # self.b_cc = Button(self.controlPanel,text="Color Code search",
        #                    width=40,pady=self.bth,border=0,bg=self.btc,
        #                    fg=self.fgc,activebackground=self.abtc,
        #                    command=lambda:self.find_distance("CC"))
        # self.b_cc.pack()
        # self.l1 = Label(self.controlPanel,bg=self.bgc,text=' ')
        # self.l1.pack()

        self.frame1 = LabelFrame(self.controlPanel,text = "Shape Feature",bg = self.bgc,bd =3)
        self.frame1.pack(fill = BOTH,expand = 1)
        self.shape_chechbox1 = Checkbutton(self.frame1,text= "Sobel",variable = self.checkbox_var[2],onvalue = 1, offvalue = 0,bg=self.bgc,padx=40)
        self.shape_chechbox1.pack(side = LEFT)
        self.shape_chechbox2 = Checkbutton(self.frame1,text= "Canny",variable = self.checkbox_var[3],onvalue = 1, offvalue = 0,bg=self.bgc,padx=45)
        self.shape_chechbox2.pack(side = RIGHT)
        # self.b_shape = Button(self.controlPanel,text="Shape code search",
        #                    width=40,border=0,pady=self.bth,bg=self.btc,
        #                    fg=self.fgc,activebackground=self.abtc,
        #                    command=lambda:self.find_distance("Shape"))
        # self.b_shape.pack()
        # self.l2 = Label(self.controlPanel,bg=self.bgc,text=' ')
        # self.l2.pack()

        self.frame3 = LabelFrame(self.controlPanel,text = "Texture Feature",bg = self.bgc,bd =3)
        self.frame3.pack(fill = BOTH,expand = 1)
        self.texture_chechbox1 = Checkbutton(self.frame3,text= "Gabor",variable = self.checkbox_var[4],onvalue = 1, offvalue = 0,bg=self.bgc,padx=40)
        self.texture_chechbox1.pack(side = LEFT)
        self.texture_chechbox2 = Checkbutton(self.frame3,text= "LBP",variable = self.checkbox_var[5],onvalue = 1, offvalue = 0,bg=self.bgc,padx=57)
        self.texture_chechbox2.pack(side = RIGHT)
        # self.b_texture = Button(self.controlPanel,text="Texture Code search",
        #                    width=40,pady=self.bth,border=0,bg=self.btc,
        #                    fg=self.fgc,activebackground=self.abtc,
        #                    command=lambda:self.find_distance("Texture"))
        # self.b_texture.pack()        
        self.l3 = Label(self.controlPanel,bg=self.bgc,text=' ')
        self.l3.pack()
        
        self.b_apply_all = Button(self.controlPanel,text="Search",
                           width=40,pady=self.bth,border=0,bg=self.btc,
                           fg=self.fgc,activebackground=self.abtc,
                           command=lambda:self.find_distance())
        self.b_apply_all.pack()        
        self.l4 = Label(self.controlPanel,bg=self.bgc,text=' ')
        self.l4.pack()

        # self.b_reset = Button(self.controlPanel,text="status",
        #                    width=40,pady=self.bth,border=0,bg=self.btc,
        #                    fg=self.fgc,activebackground=self.abtc,
        #                    command=lambda: StatsWindow(frame2))
        # self.b_reset.pack()
        # self.l5 = Label(self.controlPanel,bg=self.bgc,text=' ')
        # self.l5.pack()
        # img_pos = self.get_pos(self.selected.filename)

        # results frame

        
        self.resultFrame = Frame(self.bottomFrame,bg=self.bgc)
        self.resultFrame.pack()
        self.instr = Label(self.resultFrame,bg=self.bgc,fg='#000000',text="Precision = " + str(self.precision) + " and recall = " + str(self.recall) , font = "bold",pady=10)
        self.instr.pack()
        self.canvas = Canvas(self.resultFrame,bg=self.bgc,highlightthickness=0)

        # page navigation
        self.pageButtons = Frame(self.bottomFrame,bg=self.bgc)
        self.pageButtons.pack()
        self.b_prev = Button(self.pageButtons,text="<< Previous page",
                            width=30,border=0,bg=self.btc,
                            fg=self.fgc,activebackground=self.abtc,
                            command=lambda:self.prevPage())
        self.b_prev.pack(side=LEFT)
        self.pageLabel = Label(self.pageButtons,
                            text="Page 1 of " + str(self.totalPages),
                            width=43,bg=self.bgc,fg='#aaaaaa')
        self.pageLabel.pack(side=LEFT)
        self.b_next = Button(self.pageButtons,text="Next page >>",
                            width=30,border=0,bg=self.btc,
                            fg=self.fgc,activebackground=self.abtc,
                            command=lambda:self.nextPage())
        self.b_next.pack(side=RIGHT)        

        self.reset()


    def file_dialog(self,event=None):
        try:
            file = filedialog.askopenfilename(initialdir = os.getcwd(),title = 'Select the image',filetype = (("jpeg files","*jpg"),("all files",'*.*')))
            name = list(map(str,file.split('/')))
            f = name[-2]+"\\"+name[-1]
            self.update_preview(f)
        except:
            pass


    # resets the GUI
    def reset(self):
        self.iteration = 0
        self.weights = []
            
        # initial display photos
        self.update_preview(self.imageList[0].filename)
        self.currentImageList = self.imageList
        self.currentPhotoList = self.photoList
        il = self.currentImageList[:20]
        pl = self.currentPhotoList[:20]
        self.update_results((il, pl))

    def on_tab_selected(self,event):
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab,"text")
        if tab_text == "Histogram":
            StatsWindow()
        elif tab_text == "Preview":
            # PreView()
            self.preview()
        else:
            pass

    # find selected image position
    def get_pos(self,filename):
        pos = -1
        for i in range(len(self.imageList)):
            f=self.imageList[i].filename.replace("\\","/")
            if filename == f:
                pos = i
        return pos

    def precision_recall(self,matrix,pos):
        cnt = 0
        img_class = lambda pos : (pos//100)%100
        for i in range(len(matrix)):
            f = list(map(str,matrix[i].filename.replace("/"," ").replace("."," ").split(" ")))
            if img_class(pos) == img_class(int(f[1])):
                cnt += 1
        self.precision = cnt/20
        self.recall = cnt/100
        self.instr.configure(text = "Precision = " + str(self.precision) + " and recall = " + str(self.recall))

    # averages the feature values over size for each image
    def normalize_values(self,matrix):
        newMatrix = []
        for i in range(len(matrix)):
            x,y = self.imageList[i].size
            size = x * y
            features = [int(feat) / float(size) for feat in matrix[i]]
            newMatrix.append(features)
        return newMatrix

    def eucledian_dist(self,pos,results,*args):
        for i in range(len(args[0])):
            fd = 0
            for j in range(len(args)):
                d = 0 
                for k in range(len(args[j][0])):
                    d_i = (args[j][pos][k] - args[j][i][k])**2
                    d += d_i
                d = math.sqrt(d_i)
                fd += d
            fd = fd/len(args)
            self.insertTo(results,(fd,i),pos)

            
    # calculates eucledian distance
    def find_distance(self):
        try:
            pos = self.get_pos(self.selected.filename)
            results = []
            send = tuple()
            method_list = [self.colorBins,self.colorBinsHsv,self.shapeBins,self.shapeBinsCanny,self.textureBins,self.textureBinsLbp]
            for i in range(len(self.checkbox_var)):
                if self.checkbox_var[i].get()==1:
                    norm_vals = self.normalize_values(method_list[i])
                    send += (norm_vals,)
            self.eucledian_dist(pos,results,*send)
            
            self.currentImageList,self.currentPhotoList = [],[]
            for img in results:
                self.currentImageList.append(self.imageList[img[1]])
                self.currentPhotoList.append(self.photoList[img[1]])
            
            iL = self.currentImageList[:20]
            pL = self.currentPhotoList[:20]
            self.currentPage = 0
            self.precision_recall(iL,pos)
            self.update_results((iL,pL))
        except:
            messagebox.showerror("Feature Selection","No feature selection method is given!!")
    
    # inserts a tuple in order to arg array    
    def insertTo(self,arr,tup,pos):
        # tup[0] = distance value, [1] = image number
        img_class = lambda pos : (pos//100)%100
        if len(arr) == 0:
            arr.insert(0,tup) 
        else:
            for i in range(len(arr)):
                if tup[0] < arr[i][0]:
                    arr.insert(i,tup)
                    return
                elif img_class(tup[1])==img_class(pos) and tup[0]==arr[i][0]:
                    arr.insert(i,tup)
                    return
            arr.insert(len(arr),tup)
        return
            
     # updates the photos in results window (used from sample)   
    def update_results(self,st):
        self.pageLabel.configure(text="Page " + str(self.currentPage+1) + " of " + str(self.totalPages))
        cols = 5
        self.canvas.delete(ALL)
        self.canvas.config(width=(self.xmax)*5,height=self.ymax*4)
        self.canvas.pack()

        photoRemain = []
        for i in range(len(st[0])):
            f = st[0][i].filename
            img = st[1][i]
            photoRemain.append((f,img))
            # print(f)

        rowPos = 0
        while photoRemain:
            photoRow = photoRemain[:cols]
            photoRemain = photoRemain[cols:]
            colPos = 0
            for (filename, img) in photoRow:
                frame = Frame(self.canvas,bg=self.bgc,border=0)
                frame.pack()
                link = Button(frame,image=img,border=0,
                    bg=self.bgc,width=self.pix.get_xmax(),
                    activebackground=self.bgc)
                handler = lambda f=filename: self.update_preview(f)
                link.config(command=handler)
                link.pack(side=LEFT)
                self.canvas.create_window(
                    colPos,
                    rowPos,
                    anchor=NW,
                    window=frame, 
                    width=self.xmax, 
                    height=self.ymax)
                colPos += self.xmax
            rowPos += self.ymax

    # updates the selected image window
    def update_preview(self,f):
        self.selected = Image.open(f.replace("\\","/"))
        self.selectedPhoto=ImageTk.PhotoImage(self.selected)
        self.selectedView.configure(image=self.selectedPhoto)
        self.pos = self.get_pos(self.selected.filename)



    # updates results page to previous page
    def prevPage(self):
        self.currentPage-=1
        if self.currentPage < 0:
            self.currentPage = self.totalPages-1
        start = self.currentPage * 20
        end = start + 20
        try:
            iL = self.currentImageList[start:end]
            pL = self.currentPhotoList[start:end]
            self.update_results((iL,pL))
        except:
            pass

    # updates results page to next page
    def nextPage(self):
        self.currentPage+=1
        if self.currentPage >= self.totalPages:
            self.currentPage = 0
        start = self.currentPage * 20
        end = start + 20
        try:
            iL = self.currentImageList[start:end]
            pL = self.currentPhotoList[start:end]
            self.update_results((iL,pL))
        except:
            pass
        
    # computes total pages in results
    def get_totalPages(self):
        pages = len(self.photoList) / 20
        if len(self.photoList) % 20 > 0:
            pages += 1
        return pages

    # helper functions
    def get_selected(self):
        return self.pos
    def get_stat_frame(self):
        return self.frameStat
    def get_img_frame(self):
        return self.frameImage
    def get_photoStatList(self):
        return self.photoList
        
    #display of images in preview tab
    def preview(self):
        self.imgStat = self.frameImage
        self.img_pos = self.pos
        self.bgc = '#4e7a8a'
        self.btc = '#161616'
        self.clear_frame(self.imgStat)

        #main status frame
        self.mainStatFrame = Frame(self.imgStat,bg = self.bgc)
        self.mainStatFrame.pack(fill ="both",expand = 1)

        # view section frames
        self.topStatFrame = Frame(self.mainStatFrame,bg=self.bgc)
        self.topStatFrame.pack(side = LEFT,expand=1)
        self.middleStatFrame = Frame(self.mainStatFrame,bg=self.bgc)
        self.middleStatFrame.pack(side = LEFT,expand=1)
        self.bottomStatFrame = Frame(self.mainStatFrame,bg=self.bgc)
        self.bottomStatFrame.pack(side = LEFT,expand=1)


        self.file = "images/" + str(self.img_pos) + ".jpg"

        # reading image and converting into hsv and grayscale formate 
        self.img = cv2.imread(self.file)
        self.imgBgr = cv2.cvtColor(self.img,cv2.COLOR_RGB2BGR)
        self.imgHsv = cv2.cvtColor(self.img,cv2.COLOR_RGB2HSV)
        self.imgGray = cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY)
                    
        # Sobel shape feature extraction
        self.sobel_x = cv2.Sobel(self.imgGray,cv2.CV_64F,1,0)
        self.sobel_y = cv2.Sobel(self.imgGray,cv2.CV_64F,0,1)
        self.sobel_x = np.uint8(np.absolute(self.sobel_x))
        self.sobel_y = np.uint8(np.absolute(self.sobel_y))
        self.sobel_xy = cv2.bitwise_or(self.sobel_x,self.sobel_y)
                    
        # Canny shape feature extraction
        self.canny_img = self.auto_detect_canny(self.imgGray,0.33)
        self.canny_arr = np.array(self.canny_img)

        # Gabor texture feature extraction
        self.kernal  = cv2.getGaborKernel((5,5),5.0,1*np.pi/4,1*np.pi/4,0,ktype = cv2.CV_64F)
        self.gabor_img = cv2.filter2D(self.imgGray,cv2.CV_8UC3,self.kernal)

        # Local binary pattern texture feature extraction
        self.lbp_img = feature.local_binary_pattern(self.imgGray,8,3)

        self.tempImg = Image.fromarray(self.img)
        self.imSize = self.tempImg.size
        self.x = (self.imSize[0]-50)
        self.y = (self.imSize[1]-100)

        self.link1 = Label(self.topStatFrame,text = "BGR IMAGE",bg=self.btc,fg=self.fgc)
        self.link1.pack(side=TOP)
        self.selectedimg1 = Image.fromarray(self.imgBgr)
        self.selectedResize1 = self.selectedimg1.resize((self.x, self.y), Image.ANTIALIAS)
        self.selectedPhotoimg1 = ImageTk.PhotoImage(self.selectedResize1)
        self.frame1 = Frame(self.topStatFrame,bg=self.bgc,border=0)
        self.frame1.pack(side = TOP)
        self.link1 = Label(self.frame1,image=self.selectedPhotoimg1,bg=self.bgc)
        self.link1.pack(side=LEFT)

        self.link2 = Label(self.topStatFrame,text = "HSV IMAGE",bg=self.btc,fg=self.fgc)
        self.link2.pack(side = TOP)
        self.selectedimg2 = Image.fromarray(self.imgHsv)
        self.selectedResize2 = self.selectedimg2.resize((self.x, self.y), Image.ANTIALIAS)
        self.selectedPhotoimg2 = ImageTk.PhotoImage(self.selectedResize2)
        self.frame2 = Frame(self.topStatFrame,bg=self.bgc,border=0)
        self.frame2.pack(side = TOP)
        self.link2 = Label(self.frame2,image=self.selectedPhotoimg2,bg=self.bgc)
        self.link2.pack(side=LEFT)
        
        self.link3 = Label(self.middleStatFrame,text = "SOBEL IMAGE",bg=self.btc,fg=self.fgc)
        self.link3.pack(side=TOP)
        self.selectedimg3 = Image.fromarray(self.sobel_xy)
        self.selectedResize3 = self.selectedimg3.resize((self.x, self.y), Image.ANTIALIAS)
        self.selectedPhotoimg3 = ImageTk.PhotoImage(self.selectedResize3)
        self.frame3 = Frame(self.middleStatFrame,bg=self.bgc,border=0)
        self.frame3.pack(side = TOP)
        self.link3 = Label(self.frame3,image=self.selectedPhotoimg3,bg=self.bgc)
        self.link3.pack(side=LEFT)

        self.link4 = Label(self.middleStatFrame,text = "CANNY IMAGE",bg=self.btc,fg=self.fgc)
        self.link4.pack(side=TOP)
        self.selectedimg4 = Image.fromarray(self.canny_img)
        self.selectedResize4 = self.selectedimg4.resize((self.x, self.y), Image.ANTIALIAS)
        self.selectedPhotoimg4= ImageTk.PhotoImage(self.selectedResize4)
        self.frame4 = Frame(self.middleStatFrame,bg=self.bgc,border=0)
        self.frame4.pack(side = TOP)
        self.link4 = Label(self.frame4,image=self.selectedPhotoimg4,bg=self.bgc)
        self.link4.pack(side=LEFT)

        self.link5 = Label(self.bottomStatFrame,text = "GABOR IMAGE",bg=self.btc,fg=self.fgc)
        self.link5.pack(side=TOP)
        self.selectedimg5 = Image.fromarray(self.gabor_img)
        self.selectedResize5 = self.selectedimg5.resize((self.x, self.y), Image.ANTIALIAS)
        self.selectedPhotoimg5 = ImageTk.PhotoImage(self.selectedResize5)
        self.frame5 = Frame(self.bottomStatFrame,bg=self.bgc,border=0)
        self.frame5.pack(side = TOP)
        self.link5 = Label(self.frame5,image=self.selectedPhotoimg5,bg=self.bgc)
        self.link5.pack(side=LEFT)

        self.link6 = Label(self.bottomStatFrame,text = "LBP IMAGE",bg=self.btc,fg=self.fgc)
        self.link6.pack(side=TOP)
        self.selectedimg6 = Image.fromarray(self.lbp_img)
        self.selectedResize6 = self.selectedimg6.resize((self.x, self.y), Image.ANTIALIAS)
        self.selectedPhotoimg6 = ImageTk.PhotoImage(self.selectedResize6)
        self.frame6 = Frame(self.bottomStatFrame,bg=self.bgc,border=0)
        self.frame6.pack(side = TOP)
        self.link6 = Label(self.frame6,image=self.selectedPhotoimg6,bg=self.bgc)
        self.link6.pack(side=LEFT)

    def clear_frame(self,frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def auto_detect_canny(self,image, sigma):
        # compute the median
        mi = np.median(image)

        # computer lower & upper thresholds 
        lower = int(max(0, (1.0 - sigma) * mi))
        upper = int(min(255, (1.0 + sigma) * mi))
        image_edged = cv2.Canny(image, lower, upper)

        return image_edged

class StatsWindow(CBIR):
    def __init__(self):

        self.img_pos = top.get_selected()
        self.statFrame = top.get_stat_frame()
        self.clear_frame(self.statFrame)
        self.bgc = '#4e7a8a'

        #main status frame
        self.mainStatFrame = Frame(self.statFrame,bg = self.bgc)
        self.mainStatFrame.pack(fill ="both",expand = True)
        

        # """fetching histograms"""
        hist_rgb = pix.get_colorCode()
        hist_rgb = hist_rgb[self.img_pos]
        hist_hsv = pix.get_colorCodeHsv()
        hist_hsv = hist_hsv[self.img_pos]
        hist_sobel = pix.get_shape()
        hist_sobel = hist_sobel[self.img_pos]
        hist_canny = pix.get_shapeCanny()
        hist_canny = hist_canny[self.img_pos]
        hist_gabor = pix.get_texture()
        hist_gabor = hist_gabor[self.img_pos]
        hist_lbp = pix.get_texturelbp()
        hist_lbp = hist_lbp[self.img_pos]


        self.figure = Figure(figsize=(6,4.8), dpi=65,facecolor = self.bgc)
        self.bar = FigureCanvasTkAgg(self.figure,self.mainStatFrame)
        self.bar.get_tk_widget().pack(side=LEFT,fill ="both",expand = True)

        self.ax1 = self.figure.add_subplot(231)
        self.ax1.bar(range(12),hist_rgb,color = ['red','red','red','red','green','green','green','green','blue','blue','blue','blue'])
        self.ax1.set_xlabel("Bins")
        self.ax1.set_ylabel("Frequency")
        self.ax1.set_title('RGB Color Feature Histogram')
        self.ax1.grid()

        self.ax11 = self.figure.add_subplot(234)
        self.ax11.bar(range(12),hist_hsv,color = ['red','red','red','red','green','green','green','green','blue','blue','blue','blue'])
        self.ax11.set_xlabel("Bins")
        self.ax11.set_ylabel("Frequency")
        self.ax11.set_title('HSV Color Feature Histogram')
        self.ax11.grid()

        self.ax2 = self.figure.add_subplot(232)
        self.ax2.bar(range(12),hist_sobel)
        self.ax2.set_xlabel("Bins")
        self.ax2.set_ylabel("Frequency")
        self.ax2.set_title('Sobel Shape Feature Histogram')
        # ax2.legend()
        self.ax2.grid()

        self.ax22 = self.figure.add_subplot(235)
        self.ax22.bar(range(12),hist_canny)
        self.ax22.set_xlabel("Bins")
        self.ax22.set_ylabel("Frequency")
        self.ax22.set_title('Canny Shape Feature Histogram')
        # ax2.legend()
        self.ax22.grid()

        self.ax3 = self.figure.add_subplot(233)
        self.ax3.bar(range(12),hist_gabor)
        self.ax3.set_xlabel("Bins")
        self.ax3.set_ylabel("Frequency")
        self.ax3.set_title('Gabor Texture Feature Histogram')
        # ax2.legend()
        self.ax3.grid()

        self.ax33 = self.figure.add_subplot(236)
        self.ax33.bar(range(12),hist_lbp)
        self.ax33.set_xlabel("Bins")
        self.ax33.set_ylabel("Frequency")
        self.ax33.set_title('LBP Texture Feature Histogram')
        self.ax33.grid()

        self.figure.subplots_adjust(hspace =0.25,wspace = 0.25)
        # bar.mpl_connect('key_press_event',lambda event: self.on_key_event(event,bar,toolbar))
        # toolbar = NavigationToolbar2Tk(bar,tempframe)
        # toolbar.children['!button7'].pack_forget()
        # toolbar.update()
        # bar._tkcanvas.pack(side=TOP,fill = X)

    def clear_frame(self,frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def on_key_event(self,event,bar,toolbar):
        matplotlib.backend_bases.key_press_handler(event,bar,toolbar)

# class PreView(CBIR):
#     def __init__(self):
#         self.imgStat = top.get_img_frame()
#         self.img_pos = top.get_selected()
#         self.bgc = '#4e7a8a'
#         self.clear_frame(self.imgStat)

#         #main status frame
#         self.mainStatFrame = Frame(self.imgStat,bg = self.bgc)
#         self.mainStatFrame.pack()

#         # view section frames
#         self.topStatFrame = Frame(self.mainStatFrame,bg=self.bgc)
#         self.topStatFrame.pack(side = LEFT)
#         self.middleStatFrame = Frame(self.mainStatFrame,bg=self.bgc)
#         self.middleStatFrame.pack(side = LEFT)
#         self.bottomStatFrame = Frame(self.mainStatFrame,bg=self.bgc)
#         self.bottomStatFrame.pack(side = LEFT)


#         self.file = "images/" + str(self.img_pos) + ".jpg"

#         # reading image and converting into hsv and grayscale formate 
#         self.img = cv2.imread(self.file)
#         self.imgHsv = cv2.cvtColor(self.img,cv2.COLOR_RGB2HSV)
#         self.imgGray = cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY)
                    
        # # Sobel shape feature extraction
        # self.sobel_x = cv2.Sobel(self.imgGray,cv2.CV_64F,1,0)
        # self.sobel_y = cv2.Sobel(self.imgGray,cv2.CV_64F,0,1)
        # self.sobel_x = np.uint8(np.absolute(self.sobel_x))
        # self.sobel_y = np.uint8(np.absolute(self.sobel_y))
        # self.sobel_xy = cv2.bitwise_or(self.sobel_x,self.sobel_y)
                    
        # # Canny shape feature extraction
        # self.canny_img = self.auto_detect_canny(self.imgGray,0.33)
        # self.canny_arr = np.array(self.canny_img)

        # # Gabor texture feature extraction
        # self.kernal  = cv2.getGaborKernel((5,5),5.0,1*np.pi/4,1*np.pi/4,0,ktype = cv2.CV_64F)
        # self.gabor_img = cv2.filter2D(self.imgGray,cv2.CV_8UC3,self.kernal)

        # # Local binary pattern texture feature extraction
        # self.lbp_img = feature.local_binary_pattern(self.imgGray,8,3)

        # self.selectedimg = Image.fromarray(self.img)
        # self.imSize1 = self.selectedimg.size
        # self.x1 = self.imSize1[0]//2
        # self.y1 = self.imSize1[1]//2
        # self.selectedResize1 = self.selectedimg.resize((self.x1, self.y1), Image.ANTIALIAS)
        # self.selectedPhotoimg = ImageTk.PhotoImage(self.selectedResize1)
        # self.frame1 = Frame(self.topStatFrame,bg=self.bgc,border=0)
        # self.frame1.pack(side = BOTTOM)
        # self.link1 = Label(self.frame1,image=self.selectedPhotoimg,bg=self.bgc)
        # self.link1.pack(side=LEFT)
    
    # def auto_detect_canny(self,image, sigma):
    #     # compute the median
    #     mi = np.median(image)

    #     # computer lower & upper thresholds 
    #     lower = int(max(0, (1.0 - sigma) * mi))
    #     upper = int(min(255, (1.0 + sigma) * mi))
    #     image_edged = cv2.Canny(image, lower, upper)

    #     return image_edged

    # def clear_frame(self,frame):
    #     for widget in frame.winfo_children():
    #         widget.destroy()
        
 
class PixInfo:
    def __init__(self, master):
    
        self.master = master
        self.imageList = []
        self.photoList = []
        self.xmax = 0
        self.ymax = 0
        self.x = 0
        self.y = 0
        self.colorCode = []
        self.colorCodeHsv = []
        self.shapeHist = []
        self.shapeHistCanny = []
        self.textureHist = []
        self.textureHistLbp = []
        
        # Add each image (for evaluation) into a list (from sample code)
        for i in range(len(glob.glob('images/*.jpg'))):
            infile = "images/" + str(i) + ".jpg"
            # file, ext = os.path.splitext(infile)
            im = Image.open(infile)
            # pt = ImageTk.PhotoImage(im)

            # Resize the image for thumbnails.
            imSize = im.size
            x = imSize[0]
            y = imSize[1]
            if x > self.x:
              self.x = x
            if y > self.y:
              self.y = y
            x = imSize[0]//3
            y = imSize[1]//3

            imResize = im.resize((x, y), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(imResize)


            # Find the max height and width of the set of pics.
            if x > self.xmax:
              self.xmax = x
            if y > self.ymax:
              self.ymax = y
            # Add the images to the lists.
            self.imageList.append(im)
            self.photoList.append(photo)
        # print(len(self.imageList))  -> 1000

        # check if feature data is present or not
        if os.path.isfile('preprocessData.csv'):
            with open('preprocessData.csv','r') as df:
                csv_reader = reader(df)
                for row in csv_reader:
                    self.colorCode.append(list(map(int,row[:12])))    #RGB data from 0 t0 12
                    self.colorCodeHsv.append(list(map(int,row[12:24]))) #HSv Data from 12 to 24
                    self.shapeHist.append(list(map(int,row[24:36])))    # Sobel data from 24 to 36
                    self.shapeHistCanny.append(list(map(int,row[36:48]))) #Canny data from 36 to 48
                    self.textureHist.append(list(map(int,row[48:60])))    # Gabor data from 48 to 60
                    self.textureHistLbp.append(list(map(int,row[60:72])))  #LBP data from 60 to 72
        
        # if feature data is not present than extract feature from all individual 6 methods
        # and store it into 12 bins
        # every feature vector size wiil of 12
        # combine all feature vector to form one feature vector of size 72 (12*6)
        else:
            with open("preprocessData.csv",'w',newline="") as df:
                csv_writer= writer(df)
                for i in range(len(glob.glob('images/*.jpg'))):
                    file = "images/" + str(i) + ".jpg"

                    # reading image and converting into hsv and grayscale formate 
                    img = cv2.imread(file)
                    imgHsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
                    imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                    
                    # Sobel shape feature extraction
                    sobel_x = cv2.Sobel(imgGray,cv2.CV_64F,1,0)
                    sobel_y = cv2.Sobel(imgGray,cv2.CV_64F,0,1)
                    sobel_x = np.uint8(np.absolute(sobel_x))
                    sobel_y = np.uint8(np.absolute(sobel_y))
                    sobel_xy = cv2.bitwise_or(sobel_x,sobel_y)
                    sobel_arr = np.array(sobel_xy)
                    sobel_arr = sobel_arr.flatten()
                    
                    # Canny shape feature extraction
                    canny_img = self.auto_detect_canny(imgGray,0.33)
                    canny_arr = np.array(canny_img)
                    canny_arr = canny_arr.flatten()

                    # Gabor texture feature extraction
                    kernal  = cv2.getGaborKernel((5,5),5.0,1*np.pi/4,1*np.pi/4,0,ktype = cv2.CV_64F)
                    gabor_img = cv2.filter2D(imgGray,cv2.CV_8UC3,kernal)
                    gabor_img = gabor_img.flatten()

                    # Local binary pattern texture feature extraction
                    lbp_img = feature.local_binary_pattern(imgGray,8,3)
                    lbp_img = lbp_img.flatten()
    
                    # Creating Bins for every feature
                    rgbBins = self.new_encode(img)
                    hsvBins = self.new_encode(imgHsv)
                    sobelBins = self.mapVals(list(sobel_arr))
                    sobelBins = self.merge_bins(sobelBins)
                    cannyBins = self.mapVals(list(canny_arr))
                    cannyBins = self.merge_bins(cannyBins)
                    gaborBins = self.mapVals(list(gabor_img))
                    gaborBins = self.merge_bins(gaborBins)
                    lbpBins = self.mapVals(list(lbp_img))
                    lbpBins = self.merge_bins(lbpBins)

                    # append feature into list for processing
                    self.colorCode.append(rgbBins)
                    self.colorCodeHsv.append(hsvBins)
                    self.shapeHist.append(sobelBins)
                    self.shapeHistCanny.append(cannyBins)
                    self.textureHist.append(gaborBins)
                    self.textureHistLbp.append(lbpBins)
                    
                    # combining all bins to form single bin of size 72
                    Bins = rgbBins + hsvBins + sobelBins + cannyBins + gaborBins + lbpBins

                    csv_writer.writerow(Bins)
        
    # compute the color values
    def encode(self, pixlist):
        CcBins = [0]*64
        for inVal in pixlist:                
            r,g,b = inVal[0],inVal[1],inVal[2]
            colorValue = int(str(self.msb(r))+str(self.msb(g))+str(self.msb(b)),2)
            CcBins[colorValue] += 1             # increment bin
        return CcBins

    def new_encode(self,pixlist):
        CcBin = []
        bins_b_h,bins_g_s,bins_r_v = [0]*256,[0]*256,[0]*256
        bin_lis = (bins_b_h, bins_g_s, bins_r_v)

        d1,d2,d3 = map(lambda col : col.flatten(),cv2.split(pixlist))
        for i in range(len(d1)):
            bins_b_h[d1[i]] += 1
            bins_g_s[d2[i]] += 1
            bins_r_v[d3[i]] += 1

        for i in range(3):
            for j in range(64,257,64):
                CcBin.append(sum(bin_lis[i][j-64:j]))

        return CcBin

    def merge_bins(self,BBins):
        Bins = []
        for i in range(21,256,21):
            if i == 252:
                Bins.append(sum(BBins[i-21:]))
            else:
                Bins.append(sum(BBins[i-21:i]))
        return Bins

    # incode the texture values
    def mapVals(self,pixlist):
        TcBins = [0]*256
        for vals in pixlist:
            TcBins[int(vals)] += 1
        return TcBins
    
    # isolate the most significant 2 bits
    def msb(self, x):
        b = bin(x)              # convert to binary string
        b = b[2:]               # strip off 0b notation
        while (len(b) < 8):     # adds leading zeros
            b = '0' + b
        msb = b[:2]             # gets first 2
        return msb
    
    def auto_detect_canny(self,image, sigma):
        # compute the median
        mi = np.median(image)

        # computer lower & upper thresholds 
        lower = int(max(0, (1.0 - sigma) * mi))
        upper = int(min(255, (1.0 + sigma) * mi))
        image_edged = cv2.Canny(image, lower, upper)

        return image_edged
        
    # Accessor functions (from sample code)
    def get_imageList(self):
        return self.imageList
    def get_photoList(self):
        return self.photoList
    def get_largePL(self):
        return self.largePL
    def get_xmax(self):
        return self.xmax
    def get_ymax(self):
        return self.ymax
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def get_colorCode(self):
        return self.colorCode
    def get_colorCodeHsv(self):
        return self.colorCodeHsv
    def get_texture(self):
        return self.textureHist
    def get_texturelbp(self):
        return self.textureHistLbp
    def get_shape(self):
        return self.shapeHist
    def get_shapeCanny(self):
        return self.shapeHistCanny
    


if __name__ == '__main__':

    root = Tk()
    root.resizable(width=False, height=False)
    root.title("Context Based Image Retrival")
    pix = PixInfo(root)
    top = CBIR(root,pix)
    root.mainloop()