//
//  RFcolordetection.h
//  OPenCV test
//
//  Created by Andrew Mendez on 3/14/14.
//
//

#ifndef __OPenCV_test__RFcolordetection__
#define __OPenCV_test__RFcolordetection__

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <algorithm>    // std::max
#include <iostream>

#define PI 3.14159265


using namespace std;
using namespace cv;



 Mat src;
Mat imgIHLS;
Mat singleChannel;
Mat result;
Mat connectedImg;

int frameNum=0;

Mat trainedPoints;
Mat trainedClasses;

string winName = "meanshift";
int spatialRad, colorRad, maxPyrLevel;
Mat mImg, res;

//int lowerH=0;
//int lowerY=0;
//int lowerS=38;

int lowerH=10;
int lowerY=66;
int lowerS=35;




float mMin=2;
float mMax=-12;

//int upperH=23;
//int upperY=232;
//int upperS=156;

int upperH=23;
int upperY=255;
int upperS=255;

int lowX=10000, lowY=10000;
int highX=0,highY=0;

CvRect box;
CvRTrees  rtrees;
CvRTParams  params;

int framecount=0;


//(table.data, y=table.truth, mtry = 2, importance = FALSE, proximity = FALSE, ntree=400, do.trace = 100)


//This function threshold the HSV image and create a binary image
Mat GetThresholdedImage(Mat&src){
    Mat dst = Mat(imgIHLS.rows,imgIHLS.cols,CV_8U);
    
    inRange(src, Scalar(lowerH,lowerY,lowerS), Scalar(upperH,upperY,upperS), dst);


    
    //normalize(dst, dst, 0, 255, NORM_MINMAX, -1, CV_8UC1 );
    
    return dst;
    
}


/*
 *functions:
 */
 void RGB2HISL(Mat& src, Mat& dst){
    
    
    Vec3f result;
    for(int y=0;y<src.rows;y++){
        for(int x=0;x<src.cols;x++){
            //loop through every point in the array and
            Vec3b intensity = src.at<Vec3b>(y, x);
            int blue = intensity.val[0];
            int green = intensity.val[1];
            int red = intensity.val[2];
            
            //New image space: H,Y,S
            
            //Y(c): luminance
            result[1]=.2126*red+.7152*blue+.0722*green;
            //S(c): saturation/brightness
            result[2]=max(max(red, blue),green) - min(min(red, blue),green);
            //H(c): trignometric hue
            float t1=red - .5*green -.5*blue;
            double t2 = red*red + green*green + blue*blue - red*green - red*blue -blue*green;
            
            float t3=pow(t2, .5);
            float h;
            
            if(t3 !=0){
                
            h= acos(t1/t3);
            }
            else if(t3==0){
                h=0.0;
            }
            
            
            //calculate hue
            if(blue>green){
               
                
                float currMax=h;
                if(currMax>mMax){
                    mMax=currMax;
                   // printf("MIN: %.f\n",mMin);
                    //printf("MAX: %.f\n",mMax);
                }
                result[0]=2*PI - h;
                result[0]=result[3] * (180.0 / PI);
                
                
            }
            
            else {
                
                float currMin=h;
                if(mMin>currMin){
                    mMin=currMin;
                    //printf("MIN: %.f\n",mMin);
                    //printf("MAX: %.f\n",mMax);

                }
                //printf("MIN: %.f\n",h);

                result[0]=h * (180.0 / PI);;
                dst.at<Vec3f>(y,x) = result;
                
            }
            
            
            
            dst.at<Vec3f>(y,x) = result;
            
            
        }
    }
    
}

Vec3f bgr2IHSLconversion(int blue, int green, int red){
    
    Vec3f result;
    
    result[1]=.2126*red+.7152*blue+.0722*green;
    //S(c): saturation/brightness
    result[2]=max(max(red, blue),green) - min(min(red, blue),green);
    //H(c): trignometric hue
    float t1=red - .5*green -.5*blue;
    double t2 = red*red + green*green + blue*blue - red*green - red*blue -blue*green;
    
    float t3=pow(t2, .5);
    float h;
    
    if(t3 !=0){
        
        h= acos(t1/t3);
    }
    else if(t3==0){
        h=0.0;
    }
    
    
    //calculate hue
    if(blue>green){
        
        
        float currMax=h;
        if(currMax>mMax){
            mMax=currMax;
            // printf("MIN: %.f\n",mMin);
            //printf("MAX: %.f\n",mMax);
        }
        result[0]=2*PI - h;
        result[0]=result[3] * (180.0 / PI);
        
        
    }
    
    else {
        
        float currMin=h;
        if(mMin>currMin){
            mMin=currMin;
            //printf("MIN: %.f\n",mMin);
            //printf("MAX: %.f\n",mMax);
            
        }
        //printf("MIN: %.f\n",h);
        
        result[0]=h * (180.0 / PI);;
        return result;
        
    }
    
    
    
    
    return result;
    
    
    
    
}

//src is assumed to be a MxN 3 channel unsinged int matrix
void BGR2IHSL2(const Mat& src, Mat& dst){
    int y;
    int x;
    int c1,c2,c3;
    Vec3f result;
    for( y = 0; y < src.rows; y++)
    {
        const uchar* Mi =(const uchar*) src.ptr<uchar>(y);
        for( x = 0; x < src.cols; x++){
             c1 = (int)*Mi++;
             c2 = (int)*Mi++;
             c3 = (int)*Mi++;
        
        result = bgr2IHSLconversion(c1, c2, c3);
        dst.at<Vec3f>(y,x)=result;
        }
        
       
        
    }
        
}

void floodFillPostprocess( Mat& img, const Scalar& colorDiff=Scalar::all(1) )
{
    CV_Assert( !img.empty() );
    RNG rng = theRNG();
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            if( mask.at<uchar>(y+1, x+1) == 0 )
            {
                Scalar newVal( rng(256), rng(256), rng(256) );
                floodFill( img, mask, Point(x,y), newVal, 0, colorDiff, colorDiff );
            }
        }
    }
}
//used for meanshift segmentation
void meanShiftSegmentation( int, void* )
{
    cout << "spatialRad=" << spatialRad << "; "
    << "colorRad=" << colorRad << "; "
    << "maxPyrLevel=" << maxPyrLevel << endl;
    pyrMeanShiftFiltering( mImg, res, spatialRad, colorRad, maxPyrLevel );
    floodFillPostprocess( res, Scalar::all(2) );
    imshow( winName, res );
}
void meanShiftSegmentation(Mat& img){
    
    if( img.empty() )
        return;
    
    img.copyTo(mImg);
    
    spatialRad = 10;
    colorRad = 10;
    maxPyrLevel = 1;
    
    namedWindow( winName, CV_WINDOW_AUTOSIZE );
    
    //createTrackbar( "spatialRad", winName, &spatialRad, 80, meanShiftSegmentation );
    //createTrackbar( "colorRad", winName, &colorRad, 60, meanShiftSegmentation );
    //createTrackbar( "maxPyrLevel", winName, &maxPyrLevel, 5, meanShiftSegmentation );
    
    meanShiftSegmentation(0, 0);
    
}

int getThresholdedImage(Mat& src, Mat& dst){
    
    inRange(src, Scalar(30,34,42), Scalar(37,53,53), dst);
    
    return 0;
    
}

Mat getLaplcianEdges(Mat& src){
    
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    
    Mat gauss, gray, dst, abs_dst;
    
    
    /// Remove noise by blurring with a Gaussian filter
    GaussianBlur( src, gauss, Size(3,3), 0, 0, BORDER_DEFAULT );
    cvtColor( gauss, gray, CV_BGR2GRAY );
    //gray.copyTo(dst);
    /// Apply Laplace function
   Laplacian( gray, dst, CV_32F, kernel_size, scale, delta, BORDER_DEFAULT );
    //dst.convertTo(abs_dst, CV_8UC1);
    convertScaleAbs( dst, abs_dst );
    
    
    return abs_dst;
}

vector<int> prepare_train_classes(Mat& s){
    //normalize binary image from 255 to 1
    
    /*
     * CAN IMPROVE THIS BY DOING :
     * void convertScaleAbs(InputArray src, OutputArray dst, double,
     * alpha=1, double beta=0) BUT RESULT IN FLOAT! NEED IT TO BE IN INT
     */
    vector<int> result;
    
    for( int y = 0; y < s.rows; y ++ )
    {
        for( int x = 0; x < s.cols; x ++ )
            
        {
            int test = (int)s.at<uchar>(y,0);
            if(test==255){
                s.at<uchar>(y,x)=(uchar)(1);
            }
        }
    }
    
    Mat ans =Mat(s.rows*s.cols,1,CV_8U);
     ans = s.reshape(1, s.rows*s.cols);
    //ans.convertTo(ans, CV_8U);
    
     for( int y = 0; y < ans.rows; y++ )
     {
         int test = (int)ans.at<uchar>(y,0);
         if(test==1){
             result.push_back((int)(1));
         }
         else{
             result.push_back((int)(0));
         }
             
     //printf("%d \n", ans.at<uchar>(y, 0));
     
     }
    return ans;
}

Mat prepare_train_data(Mat& r,Mat& s, Mat &mSrc,int cond){
    Mat channel[3];
	Mat BGR;
	Mat YCbCr;
	Mat channell[3];
	Mat channelYCbCr[3];
	
    Mat h= Mat::zeros(r.rows*r.cols, cond, CV_32FC1);
    vector<cv::Mat> channels;
	
	//if (cond==7){
	
	//get BGR frame
	mSrc.copyTo(BGR);
	
	BGR.convertTo(BGR, CV_32FC3);
	//Get frame and convert to YCrCb
	mSrc.copyTo(YCbCr);
	
	cvtColor(YCbCr, YCbCr, CV_BGR2YCrCb);
	YCbCr.convertTo(YCbCr, CV_32FC3);

	
	//}
     //split image into channels
    split(r, channel);
	
	//split BGR image to add to channel
	//if (cond==7){
        split(BGR, channell);
	
	split(YCbCr, channelYCbCr);
	
	
    //}
    //convert each channel into a single column matrix-[MxN,1]
     channels.push_back(channel[0].reshape(1, channel[0].rows*channel[0].cols));
     channels.push_back(channel[1].reshape(1, channel[1].rows*channel[1].cols));
    channels.push_back(channel[2].reshape(1, channel[2].rows*channel[2].cols));
	
										   


    
    //add another column for the Laplacian edges: Mat& s
	if (cond==4) {
		channels.push_back(s.reshape(1, s.rows*s.cols));

	}
	//add another column for the Laplacian edges and another 3 columns for the Laplacian edges: Mat& mSrc

	if (cond==7) {
		
		channels.push_back(s.reshape(1, s.rows*s.cols));
		channels.push_back(channell[0].reshape(1, channell[0].rows*channell[0].cols));
    channels.push_back(channell[1].reshape(1, channell[1].rows*channell[1].cols));
		channels.push_back(channell[2].reshape(1, channell[2].rows*channell[2].cols));
		}
	
	/*
	 *add  columns for the IHSLLaplacian edges, BGR, and YCrCb

	 */
	//if (cond==10) {
		channels.push_back(s.reshape(1, s.rows*s.cols));
		//add BGR features
		channels.push_back(channell[0].reshape(1, channell[0].rows*channell[0].cols));
		channels.push_back(channell[1].reshape(1, channell[1].rows*channell[1].cols));
		channels.push_back(channell[2].reshape(1, channell[2].rows*channell[2].cols));
		//add YCbCr features
		//channels.push_back(channelYCbCr[0].reshape(1, channelYCbCr[0].rows*channelYCbCr[0].cols));
		//channels.push_back(channelYCbCr[1].reshape(1, channelYCbCr[1].rows*channelYCbCr[1].cols));
		//channels.push_back(channelYCbCr[2].reshape(1, channelYCbCr[2].rows*channelYCbCr[2].cols));
		
		
		
	//}
    
    
    //store final result in this matrix
    //Mat result =Mat(r.rows*r.cols,3,CV_32FC1);
    
    //append all columns to one matrix - matrix h
    //This will result in h being a [MxN,4] single channel matrix
    channels[0].copyTo(h.col(0));
    channels[1].copyTo(h.col(1));
     channels[2].copyTo(h.col(2));
    //Use to train with Laplacian
	if (cond==4) {
		channels[3].copyTo(h.col(3));
	}
											   
	if (cond==7) {
		channels[3].copyTo(h.col(3));
		channels[4].copyTo(h.col(4));
		channels[5].copyTo(h.col(5));
		channels[6].copyTo(h.col(6));
		
	}
	
	//add the other 7 features to the training data
	channels[3].copyTo(h.col(3));
	channels[4].copyTo(h.col(4));
	channels[5].copyTo(h.col(5));
	channels[6].copyTo(h.col(6));
	//channels[7].copyTo(h.col(7));
	//channels[8].copyTo(h.col(8));
	//channels[9].copyTo(h.col(9));
	
	
	
	
    /*
    for( int y = 0; y < h.rows; y++ )
    {
        
        printf("%f %f %f\n", h.at<float>(y, 0),h.at<float>(y, 1),h.at<float>(y, 2));
        
    }*/
    
    //printf("\n\n\n");

    //merge(channels,result);
    
    return h;
    

}
void prep_train_data( Mat& samples, Mat& classes )
{
    Mat( trainedPoints ).copyTo( samples );
    Mat( trainedClasses ).copyTo( classes );
    
    // reshape trainData and change its type
    samples = samples.reshape( 1, samples.rows );
    samples.convertTo( samples, CV_32FC1 );
}

void manualSegmentBGRfromIHSL(Mat& src,Mat& threshold){
    //This function helps visualize segmentation by checking
    //the colors of src segmented in the thresholded image
    
    for( int y = 0; y < threshold.rows; y ++ )
    {
        for( int x = 0; x < threshold.cols; x ++ )
            
        {
            int test = (int)threshold.at<uchar>(y,x);
            if (test!=255) {
                Vec3b ans;
                ans[0]=0;
                ans[1]=0;
                ans[2]=0;
                src.at<Vec3b>(y,x)=ans;
            }
            
        }
    }

}

void getBoundingBox(Mat& threshold){
    //This function helps visualize segmentation by checking
    //the colors of src segmented in the thresholded image

    for( int y = 0; y < threshold.rows; y ++ )
    {
        for( int x = 0; x < threshold.cols; x ++ )
            
        {
            int test = (int)threshold.at<uchar>(y,x);
            //printf("%d\n",test);
            if (test==255) {
                
                if(lowX>x){
                    lowX=x;
                }
                if(lowY>y){
                    lowY=y;
                }
                
                if(highX<x){
                    highX=x;
                }
                
                if(highY<y){
                    highY=y;
                }
            }
            
        }
    }
    
}

/*
 *
 */

Mat  connected_components(Mat& img){
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    int largest_area=0;
    int largest_contour_index=0;
    Rect bounding_rect;
    
    //find distance transform
    /*
   Mat dist;
    distanceTransform(img, dist, CV_DIST_L2, 3);//Get the distance transform and    
    normalize(dist, dist, 0, 1., cv::NORM_MINMAX);//normalize the result to [0,1] so we can visualize
    threshold(dist, dist, .5, 1., CV_THRESH_BINARY);//Threshold to obtain the peaks. This will be the markers for the foreground objects.
    
    // Create the CV_8U version of the distance image
    // It is needed for cv::findContours()
    cv::Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
     findContours( dist_8u, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

    */
    findContours( img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

	
	Mat dst = Mat::zeros(img.size(), CV_8UC3);
    
    if( !contours.empty() && !hierarchy.empty() )
    {
        for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
        {
            double a=contourArea( contours[i],false);  //  Find the area of contour
            if(a>largest_area){
                largest_area=a;
                largest_contour_index=i;                //Store the index of largest contour
                box=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
                Scalar color( (255), (255), (255) );
                
                drawContours( dst, contours, i, color, CV_FILLED, 8, hierarchy );
                rectangle(dst,box, color);

            }
            
        }
        
        //for( ; idx >= 0; idx = hierarchy[idx][0] )
       // {
            //Scalar color( (rand()&255), (rand()&255), (rand()&255) );
            Scalar color( (255), (255), (255) );

            //drawContours( dst, contours, largest_contour_index-2, color, CV_FILLED, 8, hierarchy );
        //}
    }
    
	return dst;
}

/* OPTOMIZED through pointer arithmetic!
 *This function checks the frame to the random forest
 * for every pixel, it tests into the random forest
 * If response is class 1 = meaning classifies pixel as hand, leaves the
 * pixel alone.
 *  else, segments out pixel of what turns out to be a failed response == 0
 */
void testRF2(int cond){
    
	Mat srcTemp;

   // Mat testSample(1, 4, CV_32FC1 );
    // test if using result from  result3.xml and below!
    // i.e. not using laplacian edges
    Mat testSample(1, cond, CV_32FC1 );
	
	cvtColor(src, srcTemp, CV_BGR2YCrCb);
    
    Mat laplacianImg=getLaplcianEdges(imgIHLS);
    
 
    
    for( int y = 0; y < imgIHLS.rows; y ++ )
    {
        
        const float* ptr =(const float*) imgIHLS.ptr<float>(y);
        for( int x = 0; x < imgIHLS.cols; x ++ )
            
        {
           // Vec3f test=imgIHLS.at<Vec3f>(y,x);
            
            testSample.at<float>(0) = *ptr++;
            testSample.at<float>(1) = *ptr++;
            testSample.at<float>(2) = *ptr++;
			if (cond==4) {
				testSample.at<float>(3) = (float)laplacianImg.at<uchar>(y,x);
			}
			
			if (cond==7) {
				testSample.at<float>(3) = (float)laplacianImg.at<uchar>(y,x);

				 Vec3b test=src.at<Vec3b>(y,x);
				testSample.at<float>(4) = (float)test[0];
				testSample.at<float>(5) = (float)test[1];
				testSample.at<float>(6) = (float)test[2];
				
			}
			
			if (cond==10) {
				testSample.at<float>(3) = (float)laplacianImg.at<uchar>(y,x);
				
				Vec3b test=src.at<Vec3b>(y,x);
				Vec3b test2=srcTemp.at<Vec3b>(y,x);
				testSample.at<float>(4) = (float)test[0];
				testSample.at<float>(5) = (float)test[1];
				testSample.at<float>(6) = (float)test[2];
				
				testSample.at<float>(7) = (float)test2[0];
				testSample.at<float>(8) = (float)test2[1];
				testSample.at<float>(9) = (float)test2[2];
				
			}
           // testSample.at<float>(3) = (float)laplacianImg.at<uchar>(y,x);
            
            
            int response = (int)rtrees.predict( testSample );
            
            //if(response==1){
               
                singleChannel.at<uchar>(y,x)=(uchar)255;
            
            //}
            
             if (response==0) {
                Vec3b ans;
                ans[0]=0;
                ans[1]=0;
                ans[2]=0;
                
                src.at<Vec3b>(y,x)=ans;
                singleChannel.at<uchar>(y,x)=(uchar)0;

            }
            
            
        }
    }
    
    //Get connected component
    
    
    
    /*
     vector<KeyPoint> corners;
     
     OrbFeatureDetector detector(700);
     detector.detect(src, corners);
     for( unsigned int i = 0; i < corners.size(); i++ )
     {
     const KeyPoint& kp = corners[i];
     circle(src, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
     }*/
    
    

}
/* This function checks the frame to the random forest
 * for every pixel, it tests into the random forest
 * If response is class 1 = meaning classifies pixel as hand, leaves the 
 * pixel alone.
 *  else, segments out pixel of what turns out to be a failed response == 0
 */
void testRF(){
    
    
    Mat testSample(1, 4, CV_32FC1 );
    // test if using result from  result3.xml and below!
    // i.e. not using laplacian edges
   // Mat testSample(1, 3, CV_32FC1 );

    
     Mat laplacianImg=getLaplcianEdges(imgIHLS);
    
    for( int y = 0; y < src.rows; y ++ )
    {
        for( int x = 0; x < src.cols; x ++ )
            
        {
            Vec3f test=imgIHLS.at<Vec3f>(y,x);
            
            testSample.at<float>(0) =test[0] ;
            testSample.at<float>(1) = test[1];
            testSample.at<float>(2) = test[2];
            //Keep to test Laplacian
            testSample.at<float>(3) = (float)laplacianImg.at<uchar>(y,x);
            
            
            int response = (int)rtrees.predict( testSample );
            
            if (response==0) {
                Vec3b ans;
                ans[0]=0;
                ans[1]=0;
                ans[2]=0;
                
                src.at<Vec3b>(y,x)=ans;
            }
            
            
        } 
    }
    /*
    vector<KeyPoint> corners;
    
   OrbFeatureDetector detector(700);
    detector.detect(src, corners);
    for( unsigned int i = 0; i < corners.size(); i++ )
    {
        const KeyPoint& kp = corners[i];
        circle(src, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
    }*/
    
    
}

void trainRF(int cond){
    /*
     * process frame
     */
    //RGB2HISL(src, imgIHLS);
    
    Mat laplacianImg=getLaplcianEdges(imgIHLS);
    
    //imshow("Laplacian", laplacianImg);
    
    imgIHLS.convertTo(imgIHLS, CV_8UC3);
    result=GetThresholdedImage(imgIHLS);
    
	/*
	 *show src image segmented from thresholded image
	 */
    manualSegmentBGRfromIHSL(src, result);
    
    vector<int> classes=prepare_train_classes(result);
    Mat mData=prepare_train_data(imgIHLS,laplacianImg,src,cond);
    
    //meanShiftSegmentation(imgIHLS);
    //printf("...\n");
    
    
    
    //mClass.convertTo( mData, CV_32SC1 );
    Mat mClass;
     Mat( classes ).copyTo( mClass );
     if(framecount==23){
     rtrees.train( mData, CV_ROW_SAMPLE, mClass, Mat(), Mat(), Mat(), Mat(), params );
     }

}

void trainRF2(int cond, Mat& src, Mat& result){
    /*
     * process frame
     */
    //RGB2HISL(src, imgIHLS);
    
    Mat laplacianImg=getLaplcianEdges(imgIHLS);
    
    //imshow("Laplacian", laplacianImg);
    
    imgIHLS.convertTo(imgIHLS, CV_8UC3);
   // result=GetThresholdedImage(imgIHLS);
    
	/*
	 *show src image segmented from thresholded image
	 */
    manualSegmentBGRfromIHSL(src, result);
    
    vector<int> classes=prepare_train_classes(result);
    Mat mData=prepare_train_data(imgIHLS,laplacianImg,src,cond);
    
    //meanShiftSegmentation(imgIHLS);
    //printf("...\n");
    
    
    
    //mClass.convertTo( mData, CV_32SC1 );
    Mat mClass;
	Mat( classes ).copyTo( mClass );
	printf("TRAINING:\n");
	//if(framecount==23){
		rtrees.train( mData, CV_ROW_SAMPLE, mClass, Mat(), Mat(), Mat(), Mat(), params );
		printf("DONE TRAINING!\n");
	//}
	
}
 int playVideo( const string source, bool testing,bool training, int cond){
     //const string source="/Users/andrewmendez/Desktop/VisionGlasses/training/truth.mp4";
     //argv[2][0]='R';
     //argv[3][0] ='N';
     /*if (argc != 4)
      {
      cout << "Not enough parameters" << endl;
      return -1;
      }*/
     
     //const string source      = argv[1];           // the source file name
     //const bool askOutputType = argv[3][0] =='Y';  // If false it will use the inputs codec type
     
     VideoCapture inputVideo(source);   // Open input
     //inputVideo.set(CV_CAP_PROP_FRAME_WIDTH, 640);
     //inputVideo.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
     if (!inputVideo.isOpened())
     {
         cout  << "Could not open the input video: " << source << endl;
         return -1;
     }
     
     string::size_type pAt = source.find_last_of('.');                  // Find extension point
     //const string NAME = source.substr(0, pAt) + argv[2][0] + ".avi";   // Form the new name with container
     int ex = static_cast<int>(inputVideo.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form
     
     // Transform from int to char via Bitwise operators
     //char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
     
     Size S = Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                   (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
     
     VideoWriter outputVideo;                                        // Open the output
      //if (askOutputType)
      //outputVideo.open("", ex=-1, inputVideo.get(CV_CAP_PROP_FPS), S, true);
	 /*
	  * SET to true to save color or 3 dimensional video
	  */
	 //outputVideo.open("/Users/andrewmendez/Desktop/VisionGlasses/training/resultvideo/fuesse2_wenigHaut_Laplacian_IHSL_BGR_YCrCb_RF.avi", ex, inputVideo.get(CV_CAP_PROP_FPS), S, true);

      //else, set false to save single channel video
      //outputVideo.open("/Users/andrewmendez/Desktop/VisionGlasses/training/resultvideo/fuesse2_wenigHaut_Laplacian_IHSL_BGR_YCrCb_RF_binary.avi", ex, inputVideo.get(CV_CAP_PROP_FPS), S, false);
     // outputVideo.op
	 /*if (!outputVideo.isOpened())
      {
      cout  << "Could not open the output video for write: " << source << endl;
      return -1;
      }*/
     
     cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
     << " of nr#: " << inputVideo.get(CV_CAP_PROP_FRAME_COUNT) << endl;
     //cout << "Input codec type: " << EXT << endl;
     
     int channel = 2; // Select the channel to save
     /*switch(argv[2][0])
      {
      case 'R' : channel = 2; break;
      case 'G' : channel = 1; break;
      case 'B' : channel = 0; break;
      }*/
     Mat res;
     //vector<Mat> spl;
     bool t=false;
     /*
     cvNamedWindow("Ball");
     
     cvCreateTrackbar("LowerH", "Ball", &lowerH, 360, NULL);
     cvCreateTrackbar("UpperH", "Ball", &upperH, 360, NULL);
     
     cvCreateTrackbar("LowerY", "Ball", &lowerY, 256, NULL);
     cvCreateTrackbar("UpperY", "Ball", &upperY, 256, NULL);
     
     cvCreateTrackbar("LowerS", "Ball", &lowerS, 256, NULL);
     cvCreateTrackbar("UpperS", "Ball", &upperS, 256, NULL);*/
     /*
     cvCreateTrackbar("LowerH", "Ball", &lowerH, 0, NULL);
     cvCreateTrackbar("UpperH", "Ball", &upperH, 256, NULL);
     
     cvCreateTrackbar("LowerS", "Ball", &lowerS, 0, NULL);
     cvCreateTrackbar("UpperS", "Ball", &upperS, 256, NULL);
     
     cvCreateTrackbar("LowerV", "Ball", &lowerV, 0, NULL);
     cvCreateTrackbar("UpperV", "Ball", &upperV, 180, NULL);
     */
     /*cvCreateTrackbar("LowerH", "Ball", &lowerH, 180, NULL);
     cvCreateTrackbar("UpperH", "Ball", &upperH, 180, NULL);
     
     cvCreateTrackbar("LowerS", "Ball", &lowerS, 256, NULL);
     cvCreateTrackbar("UpperS", "Ball", &upperS, 256, NULL);
     
     cvCreateTrackbar("LowerV", "Ball", &lowerV, 256, NULL);
     cvCreateTrackbar("UpperV", "Ball", &upperV, 256, NULL);*/

     //Mat(S.width, S.height, CV_32FC3);
          for(;;) //Show the image captured in the window and repeat
     {
         
         inputVideo >> src;
         
         
            // read
         if (src.empty()) break;         // check if at end
         if(t==false){
         src.copyTo(imgIHLS);
             //src.copyTo(result);
             result= Mat(imgIHLS.rows,imgIHLS.cols,CV_8U);
             singleChannel= Mat::zeros(src.rows,src.cols,CV_8UC1);
             

             t=true;

         }
          imgIHLS.convertTo(imgIHLS, CV_32FC3);
         //imgIHLS.copyTo(result);
         //result.convertTo(result,CV_8UC1);
         //RGB2HISL(src, imgIHLS);
         BGR2IHSL2(src, imgIHLS);

         if (testing) {
             testRF2(cond);
             connectedImg=connected_components(singleChannel);
             //connectedImg.convertTo(connectedImg, CV_8UC1);
             //connectedImg.copyTo(singleChannel);
             //getBoundingBox(singleChannel);
             //printf("%d %d %d %d\n",lowX,lowY,highX,highY);
         }
         else if (training) {
             framecount++;
             printf("framecount: %d\n",framecount);
             trainRF(cond);
         }
         
         
       
         /*
         
         
                  
          split(src, spl);                // process - extract only the correct channel
          for (int i =0; i < 3; ++i)
          if (i != channel)
          spl[i] = Mat::zeros(S, spl[0].type());
          merge(spl, res);*/
          
          //outputVideo.write(res); //save or
          //outputVideo << src;
         
         imshow("Result", connectedImg);

         
         char c=cvWaitKey(1);
         if( (char)c == 'q' )
         {
             break;
         }
         
         
     }

     
 }

int playVideo2(int cond){
	
	Mat src = imread("/Users/andrewmendez/Desktop/VisionGlasses/training/dataset training/color/scene00301.png",CV_LOAD_IMAGE_COLOR);
	Mat binary = imread("/Users/andrewmendez/Desktop/VisionGlasses/training/dataset training/binary/scene00301.png",CV_LOAD_IMAGE_GRAYSCALE);
	
	//if (src.empty()) break;         // check if at end
	//if(t==false){
		src.copyTo(imgIHLS);
		//src.copyTo(result);
		//result= Mat(imgIHLS.rows,imgIHLS.cols,CV_8U);
		singleChannel= Mat::zeros(src.rows,src.cols,CV_8UC1);
		
		
		//t=true;
		
	//}
	imgIHLS.convertTo(imgIHLS, CV_32FC3);
	//imgIHLS.copyTo(result);
	//result.convertTo(result,CV_8UC1);
	//RGB2HISL(src, imgIHLS);
	BGR2IHSL2(src, imgIHLS);
	

	trainRF2(cond,src,binary);
}





#endif /* defined(__OPenCV_test__RFcolordetection__) */
