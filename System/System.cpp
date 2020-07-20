// System.cpp : Defines the entry point for the console application.


#include "stdafx.h"

#include <opencv2/opencv.hpp>  



#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/opencv.hpp>  
#include <stdio.h>
#include <string>
#include <io.h>
#include <iostream>

using namespace cv;
using namespace std;
string nameStr[]={"Bolt","Disc","Headlift","Spring","Nut","Discringbolt"};
typedef struct MONT_LIST_S
{
	CvHuMoments hu;
	Point pt;

}MONT_LIST_T;
vector<MONT_LIST_T> detectList;
Mat training_data ; 
Mat training_classifications;
int totalNum=0;
CvRTrees* rtree ;


void printInfo()
{
	cout<<"***************************************************************"<<endl;
	cout<<"1.Feature Extraction"<<endl;
	cout<<"2.Train"<<endl;
	cout<<"3.Batching Test"<<endl;
	cout<<"4.Single Image Test"<<endl;

	cout<<"***************************************************************"<<endl;

}

int getFiles( string path, vector<string>& files )  
 {  
     long handle;                                               //handle for lookup
     struct _finddata_t fileinfo;                          //structure of file information
     handle=_findfirst(path.c_str(),&fileinfo);         //finst find
     if(handle==-1)
	 {
				 
		return -1;
	}
   //  printf("%s\n",fileinfo.name); 
	 files.push_back(fileinfo.name);
     while(!_findnext(handle,&fileinfo))               //loop through to find other matching files until no others are found
     {
		  files.push_back(fileinfo.name);
          //printf("%s\n",fileinfo.name);
     }
       _findclose(handle);                             
            
     return 0;
 }  
void CDetectSystemDlg::OnBnClickedButton2()
{
	
	int font_face = cv::FONT_HERSHEY_COMPLEX;   
	double font_scale = 2.5;  
	int thickness = 3;  

	Mat test_sample = Mat(1, 7, CV_32FC1);  
	for (int i=0;i<detectList.size();i++)
	{	
		CvHuMoments m=detectList[i].hu;
		test_sample.at<float>(0, 0) = m.hu1; 
		test_sample.at<float>(0, 1) = m.hu2; 
		test_sample.at<float>(0, 2) = m.hu3; 
		test_sample.at<float>(0, 3) = m.hu4; 
		test_sample.at<float>(0, 4) = m.hu5; 
		test_sample.at<float>(0, 5) = m.hu6; 
		test_sample.at<float>(0, 6) = m.hu7; 
		
		int  result = rtree->predict(test_sample, Mat());  
		
		AfxMessageBox(nameStr[result-1].c_str());
		putText(src,nameStr[result],detectList[i].pt,font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);


	}

	
	show_pic(&IplImage(src),IDC_SHOWPIC);

}


// Detection 
void detect(Mat &image)
{
	Mat imageGray,imageGuussian;
	Mat imageSobelX,imageSobelY,imageSobelOut;

	//1. Change original image size to improve operation efficiency
	resize(image,image,Size(500,300));
//	imshow("1",image);

	//2. Be converted into grayformat
	cvtColor(image,imageGray,CV_RGB2GRAY);
	//imshow("2.grayformt",imageGray);

	//3. The method of Gauss median-smoother filter
	GaussianBlur(imageGray,imageGuussian,Size(3,3),0);
	//imshow("3.Gauss filter",imageGuussian);

	//4.Get gradient difference between hoizontal and vertical grayscale images
	Mat imageX16S,imageY16S;
	Sobel(imageGuussian,imageX16S,CV_16S,1,0,3,1,0,4);
	Sobel(imageGuussian,imageY16S,CV_16S,0,1,3,1,0,4);
	convertScaleAbs(imageX16S,imageSobelX,1,0);
	convertScaleAbs(imageY16S,imageSobelY,1,0);
	imageSobelOut=imageSobelX-imageSobelY;
	//imshow("4.X",imageSobelX);
	//imshow("4.Y",imageSobelY);
	//imshow("4.XY",imageSobelOut);	

	//5.Mean filtering to eliminate high frequency noise
	blur(imageSobelOut,imageSobelOut,Size(3,3));
	//imshow("5.Mean filtering",imageSobelOut);	

	//6.Binary
	Mat imageSobleOutThreshold;
	threshold(imageSobelOut,imageSobleOutThreshold,180,255,CV_THRESH_BINARY);	
	//imshow("6.Binary",imageSobleOutThreshold);

	//7.Closed operation
	Mat element=getStructuringElement(0,Size(7,7));
	morphologyEx(imageSobleOutThreshold,imageSobleOutThreshold,MORPH_CLOSE,element);	
	//imshow("7.Closed operation",imageSobleOutThreshold);

	//8. Corrosion operation
	erode(imageSobleOutThreshold,imageSobleOutThreshold,element);
	//imshow("8.corrosionʴ",imageSobleOutThreshold);

	dilate(imageSobleOutThreshold,imageSobleOutThreshold,element);
	dilate(imageSobleOutThreshold,imageSobleOutThreshold,element);
	dilate(imageSobleOutThreshold,imageSobleOutThreshold,element);
	//imshow("9.",imageSobleOutThreshold);		
	vector<vector<Point>> contours;
	vector<Vec4i> hiera;

	//10. Using findcontours find the outermost contours
	int maxx=-1,maxy=-1;
	int minx=1000000,miny=1000000;
	findContours(imageSobleOutThreshold,contours,hiera,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	for(int i=0;i<contours.size();i++)
	{
		Rect rect=boundingRect((Mat)contours[i]);
		if(rect.x>maxx)
			maxx=rect.x+rect.width;
		if (rect.x<minx)
			minx=rect.x;

		if(rect.y+rect.height>maxy)
			maxy=rect.y+rect.height;
		if (rect.y<miny)
			miny=rect.y;
		//rectangle(image,rect,Scalar(255),2);	
	}	


	//circle(image,Point(maxx,maxy),3,CV_RGB(255,0,0),2,8,0);
	Rect r=Rect(minx,miny,maxx-minx,maxy-miny);

	// Mat imgroi = image(r);

	cvSaveImage("tmp.jpg",&IplImage(image));

	rectangle(image,r,Scalar(0,0,255),1);	

	//imshow("Detection results",image);

	//waitKey();
}

// Get Hu moment
int getHu(Mat &rgb,Mat &src,CvHuMoments &hu_moments)
{
	threshold(src,src,20,255,CV_THRESH_OTSU);
	Mat  element=getStructuringElement(0,Size(4,4));
	Mat  element2=getStructuringElement(0,Size(10,10));

	erode(src,src,element);

	dilate(src,src,element2);


	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;

	//Contour extraction
	findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	//Find the largest part of the image
	int w=-1;
	int idx=0;
	for (int i = 0; i < contours.size(); i++)
	{
		Rect r=boundingRect(contours[i]);
		if(r.width > w)
		{
			w=r.width;
			idx=i;
		}
	}
	if(w==-1)
	{
		return -1;
	}else {
		Moments moment=moments(contours[idx], false);
		CvMoments mon(moment);
		cvGetHuMoments(&mon, &hu_moments);

		cv::drawContours(rgb, contours, idx, cv::Scalar(0, 0, 255), 8, 8);
	}
	return 0;
}

//feature extraction
void getFeature()
{

	//seven numbers
	training_data = Mat(totalNum, 7, CV_32FC1);  
	training_classifications = Mat(totalNum, 1, CV_32FC1);  



	//obtain all files
	int startID=0;
	for ( int index = 1; index <= 6; ++index ) 
	{
		string s;
		stringstream ss;
		ss<<index;
		ss>>s;
		string path="data\\"+s+"\\*.JPG";
		vector<string> filelist;
		getFiles(path,filelist);

		for ( int i=0;i<filelist.size();i++ )
		{	
			string pp="data\\"+s+"\\"+filelist[i].c_str();
			Mat rgb=imread(pp.c_str());
			if(rgb.empty())
			{

				continue;
			}
			Mat src=imread(pp.c_str(),0);
			CvHuMoments mont;
			int ret=getHu(rgb,src,mont);


			training_classifications.at<float>(startID, 0) = index;  //class
			//Moment invariants
			training_data.at<float>(startID, 0) = mont.hu1;  
			training_data.at<float>(startID, 1) = mont.hu2;  
			training_data.at<float>(startID, 2) = mont.hu3;  
			training_data.at<float>(startID, 3) = mont.hu4;  
			training_data.at<float>(startID, 4) = mont.hu5;  
			training_data.at<float>(startID, 5) = mont.hu6;  
			training_data.at<float>(startID, 6) = mont.hu7;  

		    startID++;
			cout<<pp.c_str()<<" hu:"<<mont.hu1<<" "<<mont.hu2<<" "<<mont.hu3<<" "<<mont.hu4<<" "<<mont.hu5<<" "<<mont.hu6<<" "<<mont.hu7<<endl;
		
			namedWindow("cut",CV_WINDOW_NORMAL);
			imshow("cut",rgb);
			waitKey(1);
		}


	}

	destroyAllWindows();

}
// Implement train
void train()
{
		/********************************sStep1 Initialize Random Trees******************************/
		
		float priors[] = {1,1,1,1,1,1,1,1,1,1};  // weights of each classification for classes

		CvRTParams params = CvRTParams(25, // max depth
			5,         // min sample count
			0,        // regression accuracy: N/A here
			false,    // compute surrogate split, no missing data
			15,      // max number of categories (use sub-optimal algorithm for larger numbers)
			priors,  // the array of priors
			false,  // calculate variable importance
			4,     // number of variables randomly selected at node and used to find the best split(s).
			100,	  // max number of trees in the forest
			0.01f,// forrest accuracy
			CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
			);

		/****************************Step2 Random Decision Forest(RDF) classifiers**********************/
		Mat var_type = Mat(7 + 1, 1, CV_8U );  
		var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical  

		var_type.at<uchar>(7, 0) = CV_VAR_CATEGORICAL;

		rtree = new CvRTrees;
		rtree->train(training_data, CV_ROW_SAMPLE, training_classifications,Mat(), Mat(), var_type, Mat(), params);
		cout<<"Finished training!"<<endl;

	}



void OTSUcut(Mat &rgb,Mat &src, vector<MONT_LIST_T> &list)
{

	threshold(src,src,20,255,CV_THRESH_OTSU);


	Mat  element=getStructuringElement(0,Size(4,4));
	Mat  element2=getStructuringElement(0,Size(10,10));

	erode(src,src,element);

	dilate(src,src,element2);


	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;

	//Contour extraction
	findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));


	for (int i = 0; i < contours.size(); i++)
	{
		Rect r=boundingRect(contours[i]);
		if(r.width > 200)
		{
			MONT_LIST_T t;

			Moments moment=moments(contours[i], false);
			CvMoments mon(moment);
			cvGetHuMoments(&mon, &t.hu);
			t.pt.x=r.x+r.width/2;
			t.pt.y=r.y+r.height/2;
			list.push_back(t);


			cv::drawContours(rgb, contours, i, cv::Scalar(0, 0, 255), 8, 8);

		}
	}


}
void TestImage(const char *path)
{
	Mat src=imread(path);

	Mat gray;
	cvtColor(src,gray,CV_BGR2GRAY);

	//1.division
	detectList.clear();
	OTSUcut(src,gray,detectList);

	//2.class identification
	int font_face = cv::FONT_HERSHEY_COMPLEX;   
	double font_scale = 2.5;  
	int thickness = 3;  

	Mat test_sample = Mat(1, 7, CV_32FC1);  
	for (int i=0;i<detectList.size();i++)
	{	
		CvHuMoments m=detectList[i].hu;
		test_sample.at<float>(0, 0) = m.hu1; 
		test_sample.at<float>(0, 1) = m.hu2; 
		test_sample.at<float>(0, 2) = m.hu3; 
		test_sample.at<float>(0, 3) = m.hu4; 
		test_sample.at<float>(0, 4) = m.hu5; 
		test_sample.at<float>(0, 5) = m.hu6; 
		test_sample.at<float>(0, 6) = m.hu7; 

		int  result = rtree->predict(test_sample, Mat());  

		putText(src,nameStr[result-1],detectList[i].pt,font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);


	}
	namedWindow("cutRes",CV_WINDOW_NORMAL);
	imshow("cutRes",src);
	waitKey(0);

}



int testFun(Mat &src)
{
	int  result =-1;
	if(detectList.size()>0)
		detectList.clear();
	Mat gray;
	cvtColor(src,gray,CV_BGR2GRAY);
	OTSUcut(src,gray,detectList);

	int font_face = cv::FONT_HERSHEY_COMPLEX;   
	double font_scale = 2.5;  
	int thickness = 3;  

	Mat test_sample = Mat(1, 7, CV_32FC1);  
	for (int i=0;i<detectList.size();i++)
	{

		CvHuMoments m=detectList[i].hu;
		test_sample.at<float>(0, 0) = m.hu1; 
		test_sample.at<float>(0, 1) = m.hu2; 
		test_sample.at<float>(0, 2) = m.hu3; 
		test_sample.at<float>(0, 3) = m.hu4; 
		test_sample.at<float>(0, 4) = m.hu5; 
		test_sample.at<float>(0, 5) = m.hu6; 
		test_sample.at<float>(0, 6) = m.hu7; 

		result = rtree->predict(test_sample, Mat());  

		putText(src,nameStr[result-1],detectList[i].pt,font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);

	}
	namedWindow("res",CV_WINDOW_NORMAL);
	imshow("res",src);
	waitKey(1);
	return result;

}


void testDir()
{
	//obtain all files
	int startID=0;
	int sum=6;
	int totalImage=0;
	int alltrue=0;
	
	int *TP=new int[sum];
	int *perTotal=new int[sum];

	for(int m=0;m< 6;m++)
	{
		TP[m]=0;
		perTotal[m]=0;
		for ( int index = 0; index < 6; ++index ) 
		{

			string s;
			stringstream ss;
			ss<<index+1;
			ss>>s;	// TODO: change to the path of your testing image
			string path="test\\"+s+"\\*.JPG";
			vector<string> filelist;
			getFiles(path,filelist);

			if(m==0)
				totalImage+=filelist.size();


			for ( int i=0;i<filelist.size();i++ )
			{	
				string pp="data\\"+s+"\\"+filelist[i].c_str();
				Mat rgb=imread(pp.c_str());

				int ret=testFun(rgb);

				if(ret==(m+1)&&ret==(index+1))
					TP[m]++;


				if(ret==(m+1))
				{
					perTotal[m]++;
					alltrue++;

				}
			}
		}
   }


		/*************
		Reference:
		https://www.zhihu.com/question/30643044
		https://blog.csdn.net/u014696921/article/details/74435229

		True Positive(真正, TP) The positive class is predicted to be positive class. 

		True Negative(真负 , TN) The negative class is predicted to be negative classes

		False Positive(假正, FP) The negative class is predicted to be positive class (Type I error). 

		False Negative(假负 , FN) The positive class is predicted to be negative class (Type II error).
		***************/

		//output results
		for (int i=0;i<6;i++)
		{
			string s;
			stringstream ss;
			ss<<i+1;
			ss>>s;	// TODO: change to the path of your testing image
			string path="test\\"+s+"\\*.JPG";
			vector<string> filelist;
			getFiles(path,filelist);
			//each class
			float TPR=1.0*TP[i]/filelist.size();//tp/tp+fn TPR
			int a=perTotal[i]-TP[i];//filelist.size();
		
			int b =totalImage-filelist.size();
			float FPR=1.0*a/b;//fp/fp+tn

			//Precision tp/tp+fp
			float Precision=1.0*TP[i]/(perTotal[i]);
			//The calculation of F1, F1 = 2PR/(P+R)
			float F1=2.0*Precision*TPR/(Precision+TPR);

			printf("%s:\n",nameStr[i].c_str());
			printf("TPR:%f\nFPR:%f\nPrecision:%f\nF1:%f\n",TPR,FPR,Precision,F1);

		
			//strInfo=strInfo+nameStr[i].c_str()+":\r\n"+str;

		}
			
		printf("Correct classification:%d  InCorrect classification:%d",alltrue,totalImage-alltrue);

}


int _tmain(int argc, _TCHAR* argv[])
{
	
	for ( int index = 1; index <= 6; ++index )
	{
		string s;
		stringstream ss;
		ss<<index;
		ss>>s;    // TODO: change to the path of your testing image
		string path="data\\"+s+"\\*.JPG";
		vector<string> filelist;
		getFiles(path,filelist);
		totalNum+=filelist.size();
	}

	while(1)
	{
		printInfo();
		cout<<"Please Input:";
		int t;
		scanf("%d",&t);

		switch(t)
		{
		case 1:
			getFeature();
			break;
		case 2:
			train();
			break;
		case 3:
			testDir();
			break;
		case 4:  // TODO: change to the path of your testing image
			cout<<"Please Input Image Path:";
			string spath;
			cin>>spath;
			TestImage(spath.c_str());
			break;

		}
	}

	return 0;
}

