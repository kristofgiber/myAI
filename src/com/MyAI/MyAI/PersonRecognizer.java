package com.MyAI.MyAI;

import static  com.googlecode.javacv.cpp.opencv_highgui.*;
import static  com.googlecode.javacv.cpp.opencv_core.*;
import static  com.googlecode.javacv.cpp.opencv_imgproc.*;
import static com.googlecode.javacv.cpp.opencv_contrib.*;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.StringTokenizer;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import com.MyAI.MyAI.labels.label;
import com.googlecode.javacpp.Pointer;
import com.googlecode.javacv.cpp.opencv_core.CvFileStorage;
import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.CvMemStorage;
import com.googlecode.javacv.cpp.opencv_imgproc;
import com.googlecode.javacv.cpp.opencv_contrib.FaceRecognizer;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import com.googlecode.javacv.cpp.opencv_core.MatVector;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.content.res.XmlResourceParser;
import android.graphics.Bitmap;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

public  class PersonRecognizer {
	
	public final static int MAXIMG = 100;
	FaceRecognizer faceRecognizer;
	String mPath;
	String faceRecPath=Environment.getExternalStorageDirectory().getPath();  //getFilesDir().getPath();
	int count=0;
	labels labelsFile;
	MatVector images = new MatVector();
	int[] labels;
	//private static Context mContext;
	String rawResPath = "android.resource://com.MyAI.MyAI/"; //"android.resource://" + mContext.getPackageName() + "/";	
	
	 private int mProb=999;
	 
	 static  final double MaxPersonUncertainty = 1.07; //0.7; // EigenFaces: 0.3; FisherFaces: higher (around 0.7) because face reconstruction less accurate based on only 1 eigenvector. 
	
	 
    PersonRecognizer(String path)
    {
      //faceRecognizer =  com.googlecode.javacv.cpp.opencv_contrib.createLBPHFaceRecognizer(2,8,8,8,200);
      //faceRecognizer =  com.googlecode.javacv.cpp.opencv_contrib.createFisherFaceRecognizer();
      faceRecognizer = com.googlecode.javacv.cpp.opencv_contrib.createEigenFaceRecognizer(); // createEigenFaceRecognizer(num_components, threshold);
  	 // path=Environment.getExternalStorageDirectory()+"/facerecog/faces/";
     mPath=path;
     labelsFile= new labels(mPath);     
    }
    
    void changeRecognizer(int nRec)
    {
    	switch(nRec) {
    	case 0: faceRecognizer = com.googlecode.javacv.cpp.opencv_contrib.createLBPHFaceRecognizer(1,8,8,8,100);
    			break;
    	case 1: faceRecognizer = com.googlecode.javacv.cpp.opencv_contrib.createFisherFaceRecognizer();
    			break;
    	case 2: faceRecognizer = com.googlecode.javacv.cpp.opencv_contrib.createEigenFaceRecognizer();
    			break;
    	}
    	loadImFilesAndTrain();
    	
    }
    
	
	void addToImFiles(Mat m, String description) {		
		Bitmap bmp= Bitmap.createBitmap(m.width(), m.height(), Bitmap.Config.ARGB_8888);		 
		Utils.matToBitmap(m,bmp);
		//bmp= Bitmap.createScaledBitmap(bmp, WIDTH, HEIGHT, false);		
		FileOutputStream f;
		try {
			f = new FileOutputStream(mPath+description+"-"+count+".jpg",true);
			count++;
			bmp.compress(Bitmap.CompressFormat.JPEG, 100, f);
			f.close();
		} catch (Exception e) {
			Log.e("error",e.getCause()+" "+e.getMessage());
			e.printStackTrace();			
		}				
	}
	
	
	public boolean loadImFilesAndTrain() {	 			
		File root = new File(mPath);
        FilenameFilter pngFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase().endsWith(".jpg");            
        };
        };
        File[] imageFiles = root.listFiles(pngFilter);
        MatVector images = new MatVector(imageFiles.length);
        int[] labels = new int[imageFiles.length];
        int counter = 0;
        int label;
        IplImage img=null;
        IplImage grayImg;
        int i1=mPath.length();   
        for (File image : imageFiles) {
        	String p = image.getAbsolutePath();
            img = cvLoadImage(p);            
            if (img==null)
            	Log.e("Error","Error cVLoadImage");
            Log.i("image",p);            
            int i2=p.lastIndexOf("-");
            int i3=p.lastIndexOf(".");
            int icount=Integer.parseInt(p.substring(i2+1,i3)); 
            if (count<icount) count++;            
            String description=p.substring(i1,i2);            
            if (labelsFile.get(description)<0)
            	labelsFile.add(description, labelsFile.max()+1);            
            label = labelsFile.get(description);
            grayImg = IplImage.create(img.width(), img.height(), IPL_DEPTH_8U, 1);
            cvCvtColor(img, grayImg, CV_BGR2GRAY);
            images.put(counter, grayImg);
            labels[counter] = label;
            counter++;
        }
        if (counter>0)
        	if (labelsFile.max()>1)
        		faceRecognizer.train(images, labels);        		        		        		
        labelsFile.Save();        
		return true;
	}
	
	
	
	
	public boolean canPredict()
	{		
		if (labelsFile.max()>1)
			return true;
		else
			return false;		
	}
	
	public String predict(Mat matResized) {
		if (!canPredict())
			return "";
		int n[] = new int[1];
		double p[] = new double[1];
		//Mat matResized = new Mat();
		//Imgproc.resize(m, matResized, new Size(m.width(), m.height()));
		//IplImage ipl = MatToIplImage(m,WIDTH, HEIGHT);
		IplImage ipl = MatToIplImage(matResized,-1, -1);
		Log.e("", "n: "+n+" p: "+p);
		faceRecognizer.predict(ipl, n, p);
		
		if (n[0]!=-1){
			mProb=(int)p[0];
		}else{
			mProb=-1;
		}
	//	if ((n[0] != -1)&&(p[0]<95))
		// if (n[0] != -1)
		//matResized=IplImageToMat(ipl);
		
		//get and set threshold in runtime:
		//double current_threshold = faceRecognizer.getDouble("threshold");
		//faceRecognizer.set("threshold", current_threshold*1.1);
		
		if (verifyPerson(matResized))
			return labelsFile.get(n[0]);
		else
			return "Unkown";
	}


	/*
 	Android Bitmap's format and IplImage's format matching is very important.
	IplImage image = IplImage.create(width, height, IPL_DEPTH_8U, 4);
	IplImage _3image = IplImage.create(width, height, IPL_DEPTH_8U, 3);
	IplImage _1image = IplImage.create(width, height, IPL_DEPTH_8U, 1);
	Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);			
	1. iplimage -> bitmap
	bitmap.copyPixelsFromBuffer(image.getByteBuffer());			
	2. bitmap -> iplimage
	bitmap.copyPixelsToBuffer(image.getByteBuffer());			
	3. iplimage(4channel) -> iplimage(3channel or 1channel)
	cvCvtColor(image, _3image, CV_BGRA2BGR);
	cvCvtColor(_3image, _1image, CV_RGB2GRAY);
    */
	
	Mat CvMatToMat(CvMat inimg) {
		// bitmapToMat doesn't work on 1-layer BMP such as Config.ALPHA_8 so we need to store it in 4 layers first:
		Bitmap bmpimage=Bitmap.createBitmap(inimg.cols(), inimg.rows(), Bitmap.Config.ARGB_8888);
		bmpimage.copyPixelsFromBuffer(inimg.getByteBuffer());
		Mat matimage = new Mat();
		Utils.bitmapToMat(bmpimage, matimage);
		// return matimage;
		/// converting back to 1-layer output:
		Mat matimageGray = new Mat(inimg.cols(), inimg.rows(), CV_8UC1);
		Imgproc.cvtColor(matimage, matimageGray, CV_BGRA2GRAY); //CV_BGR2GRAY);
		return matimageGray;
	}
/*
	Mat IplImageToMat(IplImage inimg) {
		// with 1-layer input, need to create to multilayer (bitmapToMat doesn't work on 1-layer BMP such as Config.ALPHA_8):
		IplImage ipl4 = IplImage.create(inimg.width(), inimg.height(), IPL_DEPTH_8U, 4);
		cvCvtColor(inimg, ipl4, CV_GRAY2BGRA);
		Bitmap bmpimage=Bitmap.createBitmap(ipl4.width(), ipl4.height(), Bitmap.Config.ARGB_8888);
		bmpimage.copyPixelsFromBuffer(ipl4.getByteBuffer());
		Mat matimage = new Mat();
		Utils.bitmapToMat(bmpimage, matimage);
		return matimage;
		/// to 1-layer output:
		//Mat matimageGray = new Mat(ipl4.width(), ipl4.height(), CV_8UC1);
		//Imgproc.cvtColor(matimage, matimageGray, CV_BGRA2GRAY);
		//return matimageGray;
	}
*/
	  IplImage MatToIplImage(Mat m,int width,int heigth)
	  {
		 
		  
		   Bitmap bmp=Bitmap.createBitmap(m.width(), m.height(), Bitmap.Config.ARGB_8888);
		  
		   
		   Utils.matToBitmap(m, bmp);
		   return BitmapToIplImage(bmp,width, heigth);
			
	  }

	IplImage BitmapToIplImage(Bitmap bmp, int width, int height) {

		if ((width != -1) || (height != -1)) {
			Bitmap bmp2 = Bitmap.createScaledBitmap(bmp, width, height, false);
			bmp = bmp2;
		}

		IplImage image = IplImage.create(bmp.getWidth(), bmp.getHeight(),
				IPL_DEPTH_8U, 4);

		bmp.copyPixelsToBuffer(image.getByteBuffer());
		
		IplImage grayImg = IplImage.create(image.width(), image.height(),
				IPL_DEPTH_8U, 1);

		cvCvtColor(image, grayImg, opencv_imgproc.CV_BGR2GRAY);

		return grayImg;
	}	
	  
	protected void SaveBmp(Bitmap bmp,String path)
	  {
			FileOutputStream file;
			try {
				file = new FileOutputStream(path , true);
			
			bmp.compress(Bitmap.CompressFormat.JPEG,100,file); 	
		    file.close();
			}
		    catch (Exception e) {
				// TODO Auto-generated catch block
		    	Log.e("",e.getMessage()+e.getCause());
				e.printStackTrace();
			}
		
	  }
	

	public void load() {
		loadImFilesAndTrain();	
	}
	

	public void save() {
		faceRecognizer.save(mPath+"/faceRecognizer.xml"); // .yml");		
	}
	
	public int getProb() {
		// TODO Auto-generated method stub
		return mProb;
	}

	
	public boolean verifyPerson (Mat matResized) {
		CvArr eigenvectors = faceRecognizer.getMat("eigenvectors");
		CvArr averageFaceRow = faceRecognizer.getMat("mean");
		//CvMat mreshaped = new CvMat();
		//cvReshape(reconstructionRow, reconstructionMat, 1, matResized.height()); 
		IplImage img1D =MatToIplImage(matResized.reshape(1,1),-1, -1);
		// Project the input image onto the eigenspace:
		CvMat projection = subspaceProject(eigenvectors, averageFaceRow, img1D);
		// Generate the reconstructed face back from the eigenspace:
		CvMat reconstructionRow = subspaceReconstruct(eigenvectors,	averageFaceRow, projection);
		CvMat reconstructionMat = new CvMat();
		cvReshape(reconstructionRow, reconstructionMat, 1, matResized.height()); 
		Mat reconstructedFace = CvMatToMat(reconstructionMat);
		// Calculate the L2 relative error between the 2 images.
		double errorL2 = Core.norm(matResized, reconstructedFace, CV_L2);
		// Scale the value since L2 is summed across all pixels.
		double uncertainty = errorL2 / (double)(matResized.rows() * matResized.cols());
		boolean verified;
		if (uncertainty < MaxPersonUncertainty) {
			verified = true; 
			}else{
			verified = false; // Unknown person.
			}
		return verified;
	}
	
	
		
	public void LBPHfr_ReadLabelsIfEmpty(){
		if (labelsFile.max()==0){
			labelsFile.Read();
		}
	}	
		
	public void LBPHfr_loadFr() {
		//train();	
		faceRecognizer.load(mPath+"/faceRecognizer.xml");
	}
	
	void LBPHfr_addToTrainingVector(Mat m, String description, int numImgPerDetection) {
		if (count==0){
			images = new MatVector(numImgPerDetection);
			labels = new int[numImgPerDetection];
		}				
		IplImage grayImg = MatToIplImage(m,-1, -1);				
		images.put(count, grayImg);
		Log.e("","count: "+count);
		Log.e("","labels max: "+labelsFile.max());
		Log.e("","description: "+description);
        if (labelsFile.get(description)<0)
        	labelsFile.add(description, labelsFile.max()+1);
        	labelsFile.Save();
		labels[count] = labelsFile.get(description);
		count++;
	}
	
	public boolean LBPHfr_update() {		
		faceRecognizer.update(images, labels);
	return true;
	}


	
}



