package com.MyAI.MyAI;

import static com.googlecode.javacv.cpp.opencv_core.CV_L2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.StringTokenizer;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.MyAI.MyAI.R;

import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;
    
    public static final int TRAINING= 0;
    public static final int SEARCHING= 1;
    public static final int IDLE= 2;
    
    private static final int frontCam =1;
    private static final int backCam =2;
    	    		
    
    private int faceState=IDLE;
//    private int countTrain=0;
    
//    private MenuItem               mItemFace50;
//    private MenuItem               mItemFace40;
//    private MenuItem               mItemFace30;
//    private MenuItem               mItemFace20;
//    private MenuItem               mItemType;
//    
    private MenuItem               nBackCam;
    private MenuItem               mFrontCam;
    private MenuItem               mEigen;
    

    private Mat                    mRgba;
    private Mat                    mGray;					
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetect0;
    private CascadeClassifier      mJavaDetector;
    private CascadeClassifier      mJavaDetectorLeftEye;
    private CascadeClassifier      mJavaDetectorRightEye;
    private CascadeClassifier      mJavaDetectorEyeGlasses;
 //   private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;
    private int mLikely=999;
    
    private Mat m_last;
    double minFaceDelayForTraining = 1.0; //0.5;
    double minFaceChange = 0.2;
	static  final int WIDTH= 80;
	static  final int HEIGHT= 80;
    private double facechanged = 1;
    //double m_firstheight=999;
    double old_time=0;
    
    String mPath="";

    private Tutorial3View   mOpenCvCameraView;
    private int mChooseCamera = backCam;
    
    public static final boolean LOAD_PRETRAINED_MODEL = false; 
    //NOTE: LOAD_PRETRAINED_MODEL only makes sense with LBPH facerecognizers (Eigenface models have no support for .update() method and thus you could only load but not further train them)
    //If 'true': MAKE SURE to create/copy an 'assets/pretrainedFaceRec/' folder with a functioning 'pretrained_faces.xml' and a corresponding 'pretrained_faces_labels.txt' (eg. you can just copy the folder 'pretrainedFaceRec' from folder 'unused' to 'assets').
    //If 'false': DELETE folder 'pretrainedFaceRec' to avoid compiling a large unused xml.  
     
    
    EditText text;
    TextView textresult;
    private  ImageView Iv;
    Bitmap mBitmap;
    Handler mHandler;
  
    PersonRecognizer fr;
    ToggleButton toggleButtonGrabar,toggleButtonTrain,buttonSearch;
    Button buttonCatalog;
    ImageView ivGreen,ivYellow,ivRed; 
    ImageButton imCamera;
    
    TextView textState;
    com.googlecode.javacv.cpp.opencv_contrib.FaceRecognizer faceRecognizer;
   
    
    static final int MAXIMG = 5; // each detection results in 2*MAXIMG face images (normal+mirrored version)
    static final int addMirroredFaces=1; //
    static final int numImgPerDetection=MAXIMG+MAXIMG*addMirroredFaces;
	
		
    ArrayList<Mat> alimgs = new ArrayList<Mat>();

    int[] labels = new int[(int)MAXIMG];
    int countImages=0;
    
    labels labelsFile;
    
	Point leftEyeCenter;
	Point rightEyeCenter; 
    
    Resources mres;
    AssetManager assetManager;
    
    SharedPreferences mainSettings;
    SharedPreferences.Editor mainSettings_editor = null;
    boolean AssetsCopied; 
    
    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";
        Log.i(TAG, "Instantiated new " + this.getClass());
    }
    
    
    // Called when activity first created:
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.face_detect_surface_view);        
        mOpenCvCameraView = (Tutorial3View) findViewById(R.id.tutorial3_activity_java_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);       
        mainSettings = getSharedPreferences("MainSettings", 0);
        mainSettings_editor=mainSettings.edit();
        AssetsCopied = mainSettings.getBoolean("AssetsCopied",false);
        
        //mPath=getFilesDir()+"/facerecogOCV/";
        File SDCardRoot = new File(Environment.getExternalStorageDirectory().getPath()+"/MyAI");  
        if (!SDCardRoot.isDirectory()){
        	SDCardRoot.mkdirs();
        	}
        mPath=Environment.getExternalStorageDirectory().getPath()+"/myAI/";
        		
        labelsFile= new labels(mPath);                 
        Iv=(ImageView)findViewById(R.id.imageView1);
        textresult = (TextView) findViewById(R.id.textView1);        
        mHandler = new Handler() {
            @Override
            public void handleMessage(Message msg) {
            	if (msg.obj=="IMG"){
            	 Canvas canvas = new Canvas();
                 canvas.setBitmap(mBitmap);
                 Iv.setImageBitmap(mBitmap);
                 if (countImages>=MAXIMG-1){
                	 toggleButtonGrabar.setChecked(false);
                 	 grabarOnclick();
                 }
            	}
            	else{
            		String personID=msg.obj.toString();
            		textresult.setText(personID);
            		 ivGreen.setVisibility(View.INVISIBLE);
            	     ivYellow.setVisibility(View.INVISIBLE);
            	     ivRed.setVisibility(View.INVISIBLE);
            	     /*
            	    if (mLikely<0);
            	    else if (mLikely<50)
            			ivGreen.setVisibility(View.VISIBLE);
            		else if (mLikely<80)
            			ivYellow.setVisibility(View.VISIBLE);            			
            		else 
            			ivRed.setVisibility(View.VISIBLE);
					*/
            	    if (personID.equals("Unkown"))
            	    	ivRed.setVisibility(View.VISIBLE);            			
            		else 
            			ivGreen.setVisibility(View.VISIBLE);
            	}
            }
        };
        text=(EditText)findViewById(R.id.editText1);
        buttonCatalog=(Button)findViewById(R.id.buttonCat);
        toggleButtonGrabar=(ToggleButton)findViewById(R.id.toggleButtonGrabar);
        buttonSearch=(ToggleButton)findViewById(R.id.buttonBuscar);
        toggleButtonTrain=(ToggleButton)findViewById(R.id.toggleButton1);
        textState= (TextView)findViewById(R.id.textViewState);
        ivGreen=(ImageView)findViewById(R.id.imageView3);
        ivYellow=(ImageView)findViewById(R.id.imageView4);
        ivRed=(ImageView)findViewById(R.id.imageView2);
        imCamera=(ImageButton)findViewById(R.id.imageButton1);
        
        ivGreen.setVisibility(View.INVISIBLE);
        ivYellow.setVisibility(View.INVISIBLE);
        ivRed.setVisibility(View.INVISIBLE);
        text.setVisibility(View.INVISIBLE);
        textresult.setVisibility(View.INVISIBLE);
    
        toggleButtonGrabar.setVisibility(View.INVISIBLE);
        
        buttonCatalog.setOnClickListener(new View.OnClickListener() {
        	public void onClick(View view) {
        		Intent i = new Intent(com.MyAI.MyAI.FdActivity.this,
        				com.MyAI.MyAI.ImageGallery.class);
        		i.putExtra("path", mPath);
        		startActivity(i);
        	};
        	});
                
        text.setOnKeyListener(new View.OnKeyListener() {
        	public boolean onKey(View v, int keyCode, KeyEvent event) {
        		if ((text.getText().toString().length()>0)&&(toggleButtonTrain.isChecked()))
        			toggleButtonGrabar.setVisibility(View.VISIBLE);
        		else
        			toggleButtonGrabar.setVisibility(View.INVISIBLE);
        		
                return false;
        	}
        });
			        
		toggleButtonTrain.setOnClickListener(new View.OnClickListener() {
			public void onClick(View v) {
				if (toggleButtonTrain.isChecked()) {
					textState.setText(getResources().getString(R.string.SEnter));
					buttonSearch.setVisibility(View.INVISIBLE);
					textresult.setVisibility(View.VISIBLE);
					text.setVisibility(View.VISIBLE);
					textresult.setText(getResources().getString(R.string.SFaceName));
					if (text.getText().toString().length() > 0)
						toggleButtonGrabar.setVisibility(View.VISIBLE);
					
					ivGreen.setVisibility(View.INVISIBLE);
					ivYellow.setVisibility(View.INVISIBLE);
					ivRed.setVisibility(View.INVISIBLE);					

				} else {
					textState.setText(R.string.Straininig); 
					textresult.setText("");
					text.setVisibility(View.INVISIBLE);
					
					buttonSearch.setVisibility(View.VISIBLE);
					;
					textresult.setText("");
					{
						toggleButtonGrabar.setVisibility(View.INVISIBLE);
						text.setVisibility(View.INVISIBLE);
					}
			        Toast.makeText(getApplicationContext(),getResources().getString(R.string.Straininig), Toast.LENGTH_LONG).show();
			        if (LOAD_PRETRAINED_MODEL){
			        	fr.LBPHfr_update();
			        }else{
			        	fr.loadImFilesAndTrain();
			        }
					textState.setText(getResources().getString(R.string.SIdle));

				}
			}

		});
             
        toggleButtonGrabar.setOnClickListener(new View.OnClickListener() {

			public void onClick(View v) {
				grabarOnclick();
			}
		});
        
        imCamera.setOnClickListener(new View.OnClickListener() {

			public void onClick(View v) {
				
				if (mChooseCamera==frontCam)
				{
					mChooseCamera=backCam;
					mOpenCvCameraView.setCamBack();
				}
				else
				{
					mChooseCamera=frontCam;
					mOpenCvCameraView.setCamFront();
					
				}
			}
		});
        
        buttonSearch.setOnClickListener(new View.OnClickListener() {
     			public void onClick(View v) {
     				if (buttonSearch.isChecked())
     				{
     					if (!fr.canPredict())
     						{
     						buttonSearch.setChecked(false);
     			            Toast.makeText(getApplicationContext(), getResources().getString(R.string.SCanntoPredic), Toast.LENGTH_LONG).show();
     			            return;
     						}
     					textState.setText(getResources().getString(R.string.SSearching));
     					toggleButtonGrabar.setVisibility(View.INVISIBLE);
     					toggleButtonTrain.setVisibility(View.INVISIBLE);
     					text.setVisibility(View.INVISIBLE);
     					faceState=SEARCHING;
     					textresult.setVisibility(View.VISIBLE);
     				}
     				else
     				{
     					faceState=IDLE;
     					textState.setText(getResources().getString(R.string.SIdle));
     					toggleButtonGrabar.setVisibility(View.INVISIBLE);
     					toggleButtonTrain.setVisibility(View.VISIBLE);
     					text.setVisibility(View.INVISIBLE);
     					textresult.setVisibility(View.INVISIBLE);     					
     				}
     			}
     		});
        /*
        boolean success=(new File(mPath)).mkdirs();
        if (!success)
        {
        	Log.e("Error","Error creating directory");
        }
        */
    }
    
    
    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    // System.loadLibrary("detection_based_tracker");
                                 
                    fr=new PersonRecognizer(mPath);                                        
                    if (AssetsCopied){
                    	String s = getResources().getString(R.string.Straininig);
                    	Toast.makeText(getApplicationContext(),s, Toast.LENGTH_LONG).show();
                    	if (LOAD_PRETRAINED_MODEL){
                    		fr.LBPHfr_loadFr();
                    		fr.LBPHfr_ReadLabelsIfEmpty();            		    
                    		labelsFile.Read(); 
                    	}else{
                			fr.loadImFilesAndTrain();                    		
                    	}
                    }                    
                    // Copy default training data from assets:
                    mres=getResources();
                    assetManager = mres.getAssets();
                    if (!AssetsCopied){
                    	try{
	                    	if (LOAD_PRETRAINED_MODEL){
	                    		LBPHfr_copyPretrainedModel();
	                    		if (new File(mPath+"/faceRecognizer.xml").isFile()){
	                    			Toast.makeText(getApplicationContext(),"Assets copied", Toast.LENGTH_LONG).show();
	                    		}else{
	                    			Toast.makeText(getApplicationContext(),"ERROR: No pretrained face recognizer found", Toast.LENGTH_LONG).show();
	                    		}
	                    		labelsFile.Save();	       	
	                    		fr.LBPHfr_loadFr();
	                    		fr.LBPHfr_ReadLabelsIfEmpty();            		    
	                    		labelsFile.Read();   	
	                    	}else{
	                    		copyTrainingImages();	                    		
	                    		Toast.makeText(getApplicationContext(),"Assets copied", Toast.LENGTH_LONG).show();
	                    		String s = getResources().getString(R.string.Straininig);
	                            Toast.makeText(getApplicationContext(),s, Toast.LENGTH_LONG).show();
	                			fr.loadImFilesAndTrain();
	                			labelsFile.Read();	 
	                			labelsFile.Save();	       	            		    
	                    	}
	                    	AssetsCopied=true;	                    	
	                    	mainSettings_editor.putBoolean("AssetsCopied",AssetsCopied).commit();	                    	
                    	}catch(Exception e){
                    		e.printStackTrace();
                    		Log.e(TAG,"Assets first time copy failed");
                    	}
                    }
                                        
                    loadClassifier(R.raw.lbpcascade_frontalface, "lbpcascade_frontalface.xml");
                    mJavaDetector=mJavaDetect0; mJavaDetect0=null; 
                    //mNativeDetectorFace=mNativeDetector; mNativeDetector=null;
                    
                    loadClassifier(R.raw.haarcascade_mcs_lefteye, "haarcascade_mcs_lefteye.xml");
                    mJavaDetectorLeftEye=mJavaDetect0; mJavaDetect0=null; 
                    //mNativeDetectorFace=mNativeDetector; mNativeDetector=null;
                    
                    loadClassifier(R.raw.haarcascade_mcs_righteye, "haarcascade_mcs_righteye.xml");
                    mJavaDetectorRightEye=mJavaDetect0; mJavaDetect0=null; 
                    //mNativeDetectorFace=mNativeDetector; mNativeDetector=null;

                    loadClassifier(R.raw.haarcascade_eye_tree_eyeglasses, "haarcascade_eye_tree_eyeglasses.xml");
                    mJavaDetectorEyeGlasses=mJavaDetect0; mJavaDetect0=null; 
                    //mNativeDetectorFace=mNativeDetector; mNativeDetector=null;                    
                    
                    mOpenCvCameraView.enableView();
              
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
                
                
            }
        }
    };
    
    
    public void loadClassifier(int RclassifierPath, String ClassifierFileName) {
        // Load native library after(!) OpenCV initialization
        System.loadLibrary("detection_based_tracker");
        try {
            // load cascade file from application resources
            InputStream is = getResources().openRawResource(RclassifierPath);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            mCascadeFile = new File(cascadeDir, ClassifierFileName);
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            mJavaDetect0 = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            if (mJavaDetect0.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mJavaDetect0 = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

            //mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

            cascadeDir.delete();

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        	}
    };
    
    
    public void copyTrainingImages() throws Exception { 
		String facefiles[] = assetManager.list("faceimg80px");
		int fcount = 0;
		for (String filename : facefiles){
			fcount = fcount + 1;
			InputStream in = assetManager.open("faceimg80px/" + filename);					
		    OutputStream out = new FileOutputStream(new File(mPath+"/" + filename));		    
		    byte[] buffer = new byte[8192]; //[4096]
		    int read;
		    while ((read = in.read(buffer)) > 0) {
		        out.write(buffer, 0, read);
		    }		    
		    in.close();
		    out.close();		    
		    Log.i(TAG,"Built-in face image " + fcount + " copied");
		}
    };
    
    
    public void LBPHfr_copyPretrainedModel() {
		File frfile = new File(mPath+"/faceRecognizer.xml");            			
		try {			
			InputStream in = assetManager.open("pretrainedFaceRec/pretrained_faces.xml");
		    OutputStream out;
				out = new FileOutputStream(frfile);
		    // Transfer bytes from in to out
		    byte[] buf = new byte[8192]; //[4096]; //[1024];
		    int len;
		    while ((len = in.read(buf)) > 0) {
		        out.write(buf, 0, len);
		    }
		    in.close();
		    out.close();
		    Log.i(TAG,"copy SUCCESS for 'pretrained_faces.xml' face recognition file.");
		} catch (Exception e) {
			e.printStackTrace();
			Log.e(TAG,"Copy FAILED for 'assets/pretrainedFaceRec/pretrained_faces.xml'. Perhaps your .xml file content is blank or not a valid facerecognizer format.");
		}            			
		try {
			//InputStream fstream = mres.openRawResource(R.raw.pretrained_faces_labels);
			InputStream fstream = assetManager.open("pretrainedFaceRec/pretrained_faces_labels.txt");
			BufferedReader br = new BufferedReader(new InputStreamReader(fstream));
			String strLine;
			while ((strLine = br.readLine()) != null) {
				StringTokenizer tokens=new StringTokenizer(strLine,",");
				String s1=tokens.nextToken();
				String sn=tokens.nextToken();		
				labelsFile.add(s1,Integer.parseInt(sn));					
			}
			br.close();
			fstream.close();
		} catch (IOException e) {
			e.printStackTrace();
		}		    	
    };
    
    
    void grabarOnclick() {
    	if (toggleButtonGrabar.isChecked()){
			faceState=TRAINING;
    		m_last = new Mat();
    		//m_firstheight=999;
			countImages=0;
    		}else
			{ if (faceState==TRAINING)	;
			countImages=0;
			 // train();
			  //fr.loadImFilesAndTrain();
			  faceState=IDLE;
			}		
    }
    
    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        //fr.save();
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        //fr.save();
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
  	  	Imgproc.equalizeHist(mGray, mGray);
  	  	
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
          //  mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        
        MatOfRect faces = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else if (mDetectorType == NATIVE_DETECTOR) {
//            if (mNativeDetector != null)
//                mNativeDetector.detect(mGray, faces);
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

        Rect[] facesArray = faces.toArray();
        
        
        if ((facesArray.length==1)&&(faceState==TRAINING)&&(countImages<MAXIMG)&&(!text.getText().toString().isEmpty()))
        {        
       
        Mat m = new Mat();
        Mat m0= new Mat();
        Rect r = facesArray[0];
       
        
        //m=mRgba.submat(r);
        m0=mGray.submat(r);
        m=m0.clone(); // make a copy: face prediction would often crash if input was .submat (header issues with submat)

        facePreproc eyesData = new facePreproc(m);
	  	    if (eyesData.botheyesfound){
		      	//Point lefEye = eyesData.leftEyeCenter;
		      	//Point rightEye = eyesData.rightEyeCenter;	  	    	
	  	    	
	  	    	// Do all the preprocessing and resizing:
		      	m = eyesData.preprocess(m);
		      	
	  	    	// Check how long since the previous face was added.
	  	    	double current_time = (double)Core.getTickCount();
;	  	    	
	  	    	double timeDiff_seconds = 999;
	  	    	
		      	// monitor camera input for significant face changes after each snapshot:
		        if (countImages>0){// && Math.abs(1-(m.height()/m_firstheight))<0.3){
		        	//Imgproc.resize(m, m, new Size(WIDTH, HEIGHT));
		        	timeDiff_seconds = (current_time - old_time) / Core.getTickFrequency();
		        	facechanged = Core.norm(m, m_last, CV_L2) / (double)(m.rows() * m.cols());
					m_last=m;
		        }
	  	    	else{
		        	//m_firstheight=m.height();
		        	//Imgproc.resize(m, m, new Size(WIDTH, HEIGHT));
		        	m_last=m;
		        	old_time= current_time;
		        }
		        
		        
		        // if face changed enough, proceed with the training: 
				if (facechanged>minFaceChange && timeDiff_seconds>minFaceDelayForTraining){
					mBitmap = Bitmap.createBitmap(m.width(),m.height(), Bitmap.Config.ARGB_8888);
					Utils.matToBitmap(m, mBitmap);
					// SaveBmp(mBitmap,"/sdcard/db/I("+countTrain+")"+countImages+".jpg");
					Message msg = new Message();
					String textTochange = "IMG";
					msg.obj = textTochange;
					mHandler.sendMessage(msg);
					if (countImages<MAXIMG)	{
						if (LOAD_PRETRAINED_MODEL){
							fr.LBPHfr_addToTrainingVector(m, text.getText().toString(), numImgPerDetection);
						}else{
							fr.addToImFiles(m, text.getText().toString());
						}
		        		if (addMirroredFaces==1){		        			
		        			// Also add the mirror image to the training set.
		        			Mat mirroredFace = new Mat();
		        			Core.flip(m, mirroredFace, 1);
							if (LOAD_PRETRAINED_MODEL){
								fr.LBPHfr_addToTrainingVector(mirroredFace, text.getText().toString(), numImgPerDetection);
							}else{
								fr.addToImFiles(mirroredFace, text.getText().toString());
							}
		        		}
		        		countImages++;
		        	}
				}
	  	    }
        }
        else
        	 if ((facesArray.length>0)&& (faceState==SEARCHING))
          {
        	  Mat m=new Mat();
        	  Mat m0= new Mat();
        	  
        	  m0=mGray.submat(facesArray[0]);
        	  m=m0.clone(); // make a copy: face prediction would often crash if input was .submat (header issues with submat)
        	  facePreproc eyesData = new facePreproc(m);
        	  if (eyesData.botheyesfound){
	        	  //Point lefEye = eyesData.leftEyeCenter;
	        	  //Point rightEye = eyesData.rightEyeCenter;
        		  
  	  	    	  // Do all the preprocessing and resizing:
	        	  m = eyesData.preprocess(m);
	        	  
	        	  //Imgproc.resize(m, m, new Size(WIDTH, HEIGHT));
	        	  mBitmap = Bitmap.createBitmap(m.width(),m.height(), Bitmap.Config.ARGB_8888);	        	  
	             
	              Utils.matToBitmap(m, mBitmap);
	              Message msg = new Message();
	              String textTochange = "IMG";
	              msg.obj = textTochange;
	              mHandler.sendMessage(msg);
	              //Log.e("imsize: ", ""+m.size());
	              textTochange=fr.predict(m);
	              //mLikely=fr.getProb();
	        	  msg = new Message();
	        	  msg.obj = textTochange;
	        	  mHandler.sendMessage(msg);
        	  }
          }
        for (int i = 0; i < facesArray.length; i++)
            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);

        return mRgba;
    }


    
	@Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        if (mOpenCvCameraView.numberCameras()>1){
        nBackCam = menu.add(getResources().getString(R.string.SFrontCamera));
        mFrontCam = menu.add(getResources().getString(R.string.SBackCamera));
//        mEigen = menu.add("EigenFaces");
//        mLBPH.setChecked(true);
        }
        else{
        	imCamera.setVisibility(View.INVISIBLE);	        	
        }
        //mOpenCvCameraView.setAutofocus();
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
//        if (item == mItemFace50)
//            setMinFaceSize(0.5f);
//        else if (item == mItemFace40)
//            setMinFaceSize(0.4f);
//        else if (item == mItemFace30)
//            setMinFaceSize(0.3f);
//        else if (item == mItemFace20)
//            setMinFaceSize(0.2f);
//        else if (item == mItemType) {
//            mDetectorType = (mDetectorType + 1) % mDetectorName.length;
//            item.setTitle(mDetectorName[mDetectorType]);
//            setDetectorType(mDetectorType);
//        
//        }
        nBackCam.setChecked(false);
        mFrontCam.setChecked(false);
      //  mEigen.setChecked(false);
        if (item == nBackCam){
        	mOpenCvCameraView.setCamFront();
        	mChooseCamera=frontCam;
        }
        	//fr.changeRecognizer(0);
        else if (item==mFrontCam){
        	mChooseCamera=backCam;
        	mOpenCvCameraView.setCamBack();        	
        }       
        item.setChecked(true);       
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
//        if (mDetectorType != type) {
//            mDetectorType = type;
//
//            if (type == NATIVE_DETECTOR) {
//                Log.i(TAG, "Detection Based Tracker enabled");
//                mNativeDetector.start();
//            } else {
//                Log.i(TAG, "Cascade detector enabled");
//                mNativeDetector.stop();
//            }
//        }
   }
       
    
    
    public class facePreproc {
    	boolean botheyesfound;
    	Point leftEyeCenter;
    	Point rightEyeCenter;    	
    	
	    public facePreproc(Mat m) {
	    	// these are the recommended face areas for haarcascade_mcs_*eye.xml to search in:
	    	double EYE_SX = 0.10; 
	    	double EYE_SY = 0.19; 
	    	double EYE_SW = 0.40; 
	    	double EYE_SH = 0.36;
	    	//extract the recommended eye areas for eye search:
	    	int leftX = (int) Math.round(m.cols() * EYE_SX);
	    	int topY = (int) Math.round(m.rows() * EYE_SY);
	    	int widthX = (int) Math.round(m.cols() * EYE_SW);
	    	int heightY = (int) Math.round(m.rows() * EYE_SH);
	    	int rightX = (int) Math.round(m.cols() * (1.0-EYE_SX-EYE_SW));
	    	Mat topLeftOfFace = m.submat(new Rect(leftX, topY, widthX,
	    	heightY));
	    	Mat topRightOfFace = m.submat(new Rect(rightX, topY, widthX,
	    	heightY));
	    	// detect eyes in eye areas:
	    	int mAbsoluteEyeSize=(int) Math.round(mAbsoluteFaceSize*0.1);
	    	MatOfRect leftEyeRect = new MatOfRect();
	    	MatOfRect rightEyeRect = new MatOfRect();
	        mJavaDetectorLeftEye.detectMultiScale(topLeftOfFace, leftEyeRect, 1.1, 2, 2, new Size(mAbsoluteEyeSize, mAbsoluteEyeSize), new Size());
	        mJavaDetectorRightEye.detectMultiScale(topRightOfFace, rightEyeRect, 1.1, 2, 2, new Size(mAbsoluteEyeSize, mAbsoluteEyeSize), new Size());
	        // If it failed, search the left region using the glasses detector:
	    	if (leftEyeRect.toArray().length == 0)
	        	mJavaDetectorEyeGlasses.detectMultiScale(topLeftOfFace, leftEyeRect, 1.1, 2, 2, new Size(mAbsoluteEyeSize, mAbsoluteEyeSize), new Size());
	    	if (rightEyeRect.toArray().length == 0)
	        	mJavaDetectorEyeGlasses.detectMultiScale(topRightOfFace, rightEyeRect, 1.1, 2, 2, new Size(mAbsoluteEyeSize, mAbsoluteEyeSize), new Size());
	    	// Get the left eye center if one of the eye detectors worked on both sides.
	    	Rect[] LeftEye = leftEyeRect.toArray();
	    	Rect[] RightEye = rightEyeRect.toArray();
	    	leftEyeCenter = new Point(-1,-1);
	    	rightEyeCenter = new Point(-1,-1);
	    	if (LeftEye.length > 0) {
	    		leftEyeCenter.x = LeftEye[0].x + LeftEye[0].width/2 + leftX;
	    		leftEyeCenter.y = LeftEye[0].y + LeftEye[0].height/2 + topY;
	    	}
	    	if (RightEye.length > 0) {
	    		rightEyeCenter.x = RightEye[0].x + RightEye[0].width/2 + rightX;
	    		rightEyeCenter.y = RightEye[0].y + RightEye[0].height/2 + topY;
	    	}
	    	// Check if both eyes were detected.
	    	botheyesfound = false;
	    	if (leftEyeCenter.x >= 0 && rightEyeCenter.x >= 0) {
	    		botheyesfound = true;
	    	}
		}
	    	    
	    public Mat preprocess(Mat m) {	    	
	    	// Get the center between the 2 eyes.
	    	//Point2f eyesCenter = new Point2f(); // download and add vecmath-1.5.2.jar to use this class
	    	Point eyesCenter = new Point(); // Use this because Imgproc.getRotationMatrix2D() doesn't accept Point2f
	    	eyesCenter.x=(leftEyeCenter.x + rightEyeCenter.x) * 0.5f;
	    	eyesCenter.y=(leftEyeCenter.y + rightEyeCenter.y) * 0.5f;
	    	// Get the angle between the 2 eyes.
	    	double dy = (rightEyeCenter.y - leftEyeCenter.y);
	    	double dx = (rightEyeCenter.x - leftEyeCenter.x);
	    	double len = Math.sqrt(dx*dx + dy*dy);
	    	// Convert Radians to Degrees.
	    	double angle = Math.atan2(dy, dx) * 180.0/Math.PI;
	    	// Hand measurements shown that the left eye center should
	    	// ideally be roughly at (0.16, 0.14) of a scaled face image.
	    	double DESIRED_LEFT_EYE_X = 0.16;
	    	double DESIRED_LEFT_EYE_Y = 0.14; 
	    	double DESIRED_RIGHT_EYE_X = (1.0f - 0.16);
	    	// Get the amount we need to scale the image to be the desired
	    	// fixed size we want.
	    	int DESIRED_FACE_WIDTH = WIDTH;//70;
	    	int DESIRED_FACE_HEIGHT = HEIGHT;//70;
	    	double desiredLen = (DESIRED_RIGHT_EYE_X - 0.16);
	    	double scale = desiredLen * DESIRED_FACE_WIDTH / len;	    	
	    	
	    	// Get the transformation matrix for the desired angle & size.
	    	Mat rot_mat = Imgproc.getRotationMatrix2D(eyesCenter, angle, scale);
	    	// Shift the center of the eyes to be the desired center.
	    	double ex = DESIRED_FACE_WIDTH * 0.5f - eyesCenter.x;
	    	double ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter.y;
	    	rot_mat.put(0, 2, rot_mat.get(0, 2)[0] + ex); //rot_mat.at<double>(0, 2) += ex;
	    	rot_mat.put(1, 2, rot_mat.get(1, 2)[0] + ey); //rot_mat.at<double>(1, 2) += ey;	    		    	
	    	// Transform the face image to the desired angle & size &
	    	// position! Also clear the transformed image background to a
	    	// default grey.
	    	Mat warped = new Mat(DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH, CvType.CV_8U, new Scalar(128));
	    	Imgproc.warpAffine(m, warped, rot_mat, warped.size());
			m=warped;
			
	    	// Apply histogram equalization left and right side separately and then smooth the midline edge by filtering:
	    	int w = m.cols();
	    	int h = m.rows();
	    	Mat wholeFace = new Mat();
	    	Imgproc.equalizeHist(m, wholeFace);
	    	int midX = w/2;
	    	Mat leftSide = m.submat(new Rect(0,0, midX,h));
	    	Mat rightSide = m.submat(new Rect(midX,0, w-midX,h));
	    	Imgproc.equalizeHist(leftSide, leftSide);
	    	Imgproc.equalizeHist(rightSide, rightSide);
	    	
	    	for (int y=0; y<h; y++) {
	    		for (int x=0; x<w; x++) {
		    		if (x < w/4) {
		    		// Left 25%: just use the left face.
		    		//m.put(y,x,leftSide.get(y,x));   		
		    		}
		    		else if (x < w*2/4) {
		    		// Mid-left 25%: blend the left face & whole face.
		    		int lv = (int) leftSide.get(y, x)[0];
		    		int wv = (int) wholeFace.get(y, x)[0];
		    		// Blend more of the whole face as it moves
		    		// further right along the face.
		    		float f = (x - w*1/4) / (float)(w/4);
		    		m.put(y,x,Math.round((1.0f - f) * lv + (f) * wv));
		    		}
		    		else if (x < w*3/4) {
		    		// Mid-right 25%: blend right face & whole face.
		    		int rv = (int) rightSide.get(y,x-midX)[0];
		    		int wv = (int) wholeFace.get(y, x)[0];
		    		// Blend more of the right-side face as it moves
		    		// further right along the face.
		    		float f = (x - w*2/4) / (float)(w/4);
		    		m.put(y,x,Math.round((1.0f - f) * wv + (f) * rv));
		    		}
		    		else {
		    		// Right 25%: just use the right face.
		    		m.put(y,x,rightSide.get(y,x-midX));
		    		}
	    		}
	    	}
	    	// smoothing noise:
	    	Mat filtered = new Mat(m.size(), CvType.CV_8U);
	    	Imgproc.bilateralFilter(m, filtered, 0, 20.0, 2.0);	    	
	    	
	    	// Draw a black-filled ellipse in the middle of the image.
	    	// First we initialize the mask image to white (255).
	    	Mat mask = new Mat(m.size(), CvType.CV_8UC1, new Scalar(255));
	    	double dw = DESIRED_FACE_WIDTH;
	    	double dh = DESIRED_FACE_HEIGHT;
	    	Point faceCenter = new Point( Math.round(dw * 0.5), Math.round(dh * 0.4));
	    	Size size = new Size( Math.round(dw * 0.5), Math.round(dh * 0.8) );
	    	Core.ellipse(mask, faceCenter, size, 0, 0, 360, new Scalar(0), -1);	    	
	    	// Apply the elliptical mask on the face, to remove corners.
	    	// Sets corners to gray, without touching the inner face.
	    	filtered.setTo(new Scalar(128), mask);
	    	m=filtered;
	    	
			return m;
	    }
	    
    }

    

}
