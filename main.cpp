#include "opencv2/aruco.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/ccalib.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

#include <sstream>
#include <iostream>
#include <fstream>


using namespace std;

using namespace cv;
const int fps = 30;
const float caliSquareDimension = 0.02f;
const float arucoSquareDimension = 0.015f;
const Size dimensions = Size(9,7);

void createArucoMarkers() {
	
	Mat outputMarker;

	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

	for (int i = 0; i < 50; i++) {
		aruco::drawMarker(markerDictionary, i, 500, outputMarker,1);
		ostringstream convert;
		string imageName = "4x4Marker_";
		convert << imageName << i << ".jpg";
		imwrite(convert.str(), outputMarker);
	}
}

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners) {
	for (int i = 0; i < boardSize.height;  i++)
	{
		for (int j = 0; j < boardSize.width; j++)
		{
			corners.push_back(Point3f(j*squareEdgeLength, i*squareEdgeLength, 0.0f));

		}
	
	}
}
void getChessCorners(vector <Mat> images,vector <vector<Point2f>>& allFoundCorners, bool showResult =false) {
	
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++) {
		vector<Point2f>pointBuf;
		bool found = findChessboardCorners(*iter, dimensions, pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
		if (found) {
			allFoundCorners.push_back(pointBuf);
		 }
		if (showResult) {
			drawChessboardCorners(*iter, dimensions, pointBuf, found);
			imshow("lookup",*iter);
			waitKey(0);
		}
	}
}

void cameraCalib(vector<Mat> calibImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat distanceCoefficeients) {
	
	vector<vector<Point2f>>checkerboardImagesSpacePoints;
	getChessCorners(calibImages, checkerboardImagesSpacePoints, false);
	vector<vector<Point3f>> worldSpaceCornerPoints(1);
	createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
	worldSpaceCornerPoints.resize(checkerboardImagesSpacePoints.size(), worldSpaceCornerPoints[0]);
	vector<Mat> rVectors, tVectors;
	distanceCoefficeients = Mat::zeros(8, 1, CV_64F);
	calibrateCamera(worldSpaceCornerPoints, checkerboardImagesSpacePoints, boardSize, cameraMatrix, distanceCoefficeients, rVectors, tVectors);
}
bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients) {
	ofstream outstream(name);
	
	if (outstream) {
		
		uint16_t rows = cameraMatrix.rows;
		uint16_t cols = cameraMatrix.cols;

		outstream << rows << endl;
		outstream << cols << endl;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				
				double value = cameraMatrix.at<double>(r, c);
				cout << value;
				outstream << value << endl;
			}

		}
		rows = distanceCoefficients.rows;
		cols = distanceCoefficients.cols;
		outstream << rows << endl;
		outstream << cols << endl;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				double value = distanceCoefficients.at<double>(r, c);
				outstream << value << endl;
			}

		}
		outstream.close();
		cout << "should done?";
		return true;
	}
	cout << "some Fail";
	return false;
	
}
bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients) {
	ifstream inStream(name);
	if (inStream) {
		uint16_t rows;
		uint16_t cols;

		inStream >> rows;
		inStream >> cols;

		cameraMatrix = Mat(Size(cols, rows), CV_64F);

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				double read = 0.0f;
				inStream >> read;
				cameraMatrix.at<double>(r, c) = read;
				cout << cameraMatrix.at<double>(r, c) << "\n ";
			}

		}
		//distCoefficient
		inStream >> rows;
		inStream >> cols;

		distanceCoefficients = Mat::zeros(rows, cols,CV_64F);
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				double read = 0.0f;
				inStream >> read;
				distanceCoefficients.at<double>(r, c) = read;
				cout << distanceCoefficients.at<double>(r, c) << "\n ";
			}

		}
		inStream.close();
		return true;
	}
	return false;

}
int startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distenceCoeffiecints, float arucoSquareDimensions) {
	Mat frame;
	vector<int> markerIds;
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	aruco::DetectorParameters parameters;
	cv::Mat Rot(3, 3, CV_32FC1);
	Ptr < aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);
	VideoCapture vid(0);

	if (!vid.isOpened()) {
		return -1;

	}
	namedWindow("Webcam", WINDOW_AUTOSIZE);
	vector<Vec3d > rotationVector, translationVector;

	while (true) {

		if (!vid.read(frame)) {
			break;
		}
		aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
		aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
		aruco::estimatePoseSingleMarkers(markerCorners,arucoSquareDimension, cameraMatrix ,distenceCoeffiecints, rotationVector,translationVector);
		
			for (int i = 0; i < markerIds.size(); i++)
			{
				aruco::drawAxis(frame, cameraMatrix, distenceCoeffiecints, rotationVector[i],translationVector[i],0.1f);
				
			}
			imshow("Webcam",frame);
			if (waitKey(30) >= 0)break;

	}

	return 1;

}

void cameraCalibrationProcess(Mat& cameraMatrix, Mat& distanceCoefficients) {
	Mat frame;
	Mat drawToFrame;
	

	vector<Mat> savedImages;
	vector<vector<Point2f>>markerCorners, rejectedCan;

	VideoCapture vid(0);

	if (!vid.isOpened()) {
		return;
	}
	int fps = 20;
	namedWindow("Webcam", WINDOW_AUTOSIZE);
	while (true) {
		if (!vid.read(frame))
			break;

		vector<Vec2f>foundPoints;
		bool found = false;

		found = findChessboardCorners(frame, dimensions, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		frame.copyTo(drawToFrame);
		drawChessboardCorners(drawToFrame, dimensions, foundPoints, found);

		if (found)
			imshow("Webcam", drawToFrame);
		else
			imshow("Webcam", frame);

		char character = waitKey(1000 / fps);

		switch (character)
		{
		case 32:
			//save an Image
			cout << "should save";
			if (found) {
				Mat temp;
				frame.copyTo(temp);
				savedImages.push_back(temp);
			}
			break;
		case 13:
			//start calibaration
			cout << "should out";
			if (savedImages.size() > 2) {
				cameraCalib(savedImages, dimensions, caliSquareDimension, cameraMatrix, distanceCoefficients);
				saveCameraCalibration("camCalib", cameraMatrix, distanceCoefficients);
			}
			break;
		case 27:
			//stooppp
			return;
			break;
		}
	}

}
int main(int argv, char** argc) {
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	Mat distanceCoefficients;
	//cameraCalibrationProcess(cameraMatrix, distanceCoefficients);
	loadCameraCalibration("camCalib", cameraMatrix,distanceCoefficients);
	startWebcamMonitoring(cameraMatrix, distanceCoefficients, arucoSquareDimension);

	return 0;
}