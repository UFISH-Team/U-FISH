//// 2D grid search script. ////
/// Copied RS-FISH team some code ///

// Input images path
dir = "ufish/valid/image-uint8/";

// Output csv path
csv = "ufish/valid/RS_results/"

setBatchMode(true);

totalStartTime = getTime();

//////// Define grid search parameters: //////////
// start, end, step (inclusive)
sig_range = newArray(1,2,0.25); // 4 steps
thr_range_xstep = newArray(0.003,0.08,1.50); // 8 steps - multiplication!
suppReg_range = newArray(2,4,1); // 3 steps
intenThr_vals = newArray(0,10,50,100,150,200,255); // 8 steps // thats the actual array

//////// Use multithreading: //////////
useMultithread = "use_multithreading";	// If you wish to use multithreading "use_multithreading", else "" (empty string)
numThreads = 16;						// multithread param
blockSizX = 128;                     	// multithread param
blockSizY = 128;						// multithread param
blockSizZ = 1;							// multithread param
///////////////////////////////////////////////////

// Location of file with all the run times that will be saved:
timeFile = "ufish/benchmarks/RS/RS_" + dir_name + "_exeTime.txt";

walkFiles(dir);

// Find all files in subdirs:
function walkFiles(dir) {
	list = getFileList(dir);
	for (i=0; i<list.length; i++) {
		if (endsWith(list[i], "/"))
		   walkFiles(""+dir+list[i]);

		// If image file
		else  if (endsWith(list[i], ".tif"))
		   gridProcessImage(dir, csv, list[i]);
	}
}

function gridProcessImage(dirPath, csvPath, imName) {

	open("" + dirPath + imName);

	//// GRID SEARCH
	aniso = 1;
	inlierRatio = 0.1;
	maxError = 1.5000;

	for (sig=sig_range[0]; sig<=sig_range[1]; sig=sig+sig_range[2]) {
		for (thr=thr_range_xstep[0]; thr<=thr_range_xstep[1]; thr=thr*thr_range_xstep[2]) {
			for (suppReg=suppReg_range[0]; suppReg<=suppReg_range[1]; suppReg=suppReg+suppReg_range[2]) {
				//for (inRat=inRat_range[0]; inRat<=inRat_range[1]; inRat=inRat+inRat_range[2]) {
					//for (maxErr=maxErr_range[0]; maxErr<=maxErr_range[1]; maxErr=maxErr+maxErr_range[2]) {
						for (iit=0; iit<intenThr_vals.length; iit++) {
							
							intesThr = intenThr_vals[iit];

					        results_csv_path = "" + csvPath + "RS_" + imName  +
					        "_sig" + sig +
					        "thr" + thr +
					        "suppReg" + suppReg +
					        "intensThr" + intesThr +
					        ".csv";

					        RSparams = "image=" + imName +
					        " mode=Advanced anisotropy=" + aniso + " use_anisotropy" +
					        " robust_fitting=RANSAC" +
					        " sigma=" + sig +
					        " threshold=" + thr +
					        " support=" + suppReg +
				        	" spot_intensity_threshold=" + intesThr +
				        	" background=[No background subtraction]" +
				        	" results_file=[" + results_csv_path + "]";
				        	//" " + useMultithread + " num_threads=" + numThreads + " block_size_x=" + blockSizX + " block_size_y=" + blockSizY + " block_size_z=" + blockSizZ;

				        	print(RSparams);
				            startTime = getTime();
				        	run("RS-FISH", RSparams);
					        exeTime = getTime() - startTime; //in miliseconds

					         // Save exeTime to file:
					        File.append(results_csv_path + "," + exeTime + "\n ", timeFile);
						}
					//}
				//}
			}
		}
	}
	// Close all windows:
	run("Close All");	
	while (nImages>0) { 
		selectImage(nImages); 
		close(); 
    } 
} 

totalEndTime = getTime();
totalExeTime = totalEndTime - totalStartTime; # in miliseconds

print("Total execution time: " + totalExeTime + " ms");

// Save totalExeTime to file:
File.append(timeFile, "Total execution time: " + totalExeTime + " ms\n");
