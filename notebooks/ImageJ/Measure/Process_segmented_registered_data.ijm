 /*
 * This script measures all relevant microenvironment paramters and allows for on-the-fly inspection
 */

inspect = true;
start_at = 17;
stop_at = 44;

// config
run("CLIJ2 Macro Extensions", "cl_device=[GeForce RTX 3060 Ti]");
Ext.CLIJ2_clear();
roiManager("reset");


if (inspect) {
	setBatchMode(false);
} else {
	setBatchMode(true);	
}

if (inspect) {
	run("Table...", "name=[Measurements] width=800 height=600");
	print("[Measurements]", "\\Headings:Name\	Necrotic_fraction\	CD31_fraction\	Perfused_fraction\	" +
							"Pimo_Area\	Vessel_radius_median\	Hypoxic Distance\	Rating seg\	Comment");
} else {
	run("Table...", "name=[Measurements] width=800 height=600");
	print("[Measurements]", "\\Headings:Name\	Necrotic_fraction\	CD31_fraction\	Perfused_fraction\	" +
							"Pimo_Area\	Vessel_radius_median\	Hypoxic Distance");
}

var PixelSize = 0.442;  // microns

// clean up
close("*");

smoothing_vital = 6;
smoothing_mask = 10;

// Variables
var Mask_Tumor = "Mask_Tumor";
var Mask_Vital = "Mask_Vital";

// Paths
root = "E:/Promotion/Projects/2020_Radiomics/Data/";

// Input dir
mice = getsubdirs(root);

// iterate over all mice
for (i = start_at; i < mice.length; i++) {


	dir_sample = root + mice[i] + "Histology/";
	samples = getsubdirs(dir_sample);
	print(d2s(i+1, 0) + "/" + d2s(mice.length, 0) + " animals: " + dir_sample);

	// iterate over samples (aka tissue sections)
	for (j = 0; j < samples.length; j++) {
		Ext.CLIJ2_clear();
		close("*");
		roiManager("reset");
		seg_IF = false;
		seg_HE = false;
		reg = false;
		included = true;

		dir_raw = dir_sample + samples[j];
		dir_seg = dir_sample + samples[j] + "1_seg/";
		dir_reg = dir_sample + samples[j] + "2_reg/";
		dir_meas = dir_sample + samples[j] + "3_meas/";

		if (!File.exists(dir_meas)) {
			File.makeDirectory(dir_meas);
		}

		// has file been censored?
		if (matches(samples[j], ".*censored.*")) {
			included = false;
			continue;
		}

		// All necessary files present?
		if (File.exists(dir_seg + "HE_seg_DL.tif")) {
			seg_HE = true;
		}

		if (File.exists(dir_seg + "IF_seg_Simple_Segmentation.tif")) {
			seg_IF = true;
		}

		if (File.exists(dir_reg + "IF_seg_Simple_Segmentation_transformed.tif")) {
			reg = true;
		}

		checksum = seg_HE + seg_IF + reg + included;
		if (checksum == 4) {
			print("    ---> Checksum: ", seg_HE, seg_IF, reg, included);
			print("    ---> All necessary input detected");
		} else {
			print("    ---> Missing data. Checksum: ", seg_HE, seg_IF, reg, included);
			continue;
		}

		// Move data to measurement dir
		fname_HE_old = dir_seg + "HE_seg_Simple_Segmentation.tif";
		fname_HE = dir_seg + "HE_seg_DL.tif";
		fname_IF = dir_reg + "IF_seg_Simple_Segmentation_transformed.tif";

		// Step 0: Import BlackTiles.zip
		roiManager("open", dir_seg + "BlackTiles.zip");

		// Step 1: Import transformed IF image and convert to propper format
		open(fname_IF);
		IF = getTitle();

		// Step 2: 
		// Use cut-off and upsampling on UNet output image
		run("Bio-Formats Importer", "open="+fname_HE);
		HE_probabilityMap = getTitle();
		VitalCutOff(HE_probabilityMap, IF);
		exit();
		IF = procIF(IF);

		// inspect images
		/*
		if (inspect) {
			f = findFileByID(dir_raw, "HE");
			run("Bio-Formats Importer", "open="+dir_raw + f[0]+" autoscale series_1");
			HE_raw = getTitle();
			inspect_raw_and_seg(newArray("Mask_Tumor", "Mask_Vital", HE_raw), 0, false, true);

			f_IF_raw = findFileByID(dir_raw, "IF");
			run("Bio-Formats Importer", "open="+dir_raw + f_IF_raw[0]+" autoscale series_1");
			IF_raw = getTitle();
			opinion = inspect_raw_and_seg(newArray("Mask_Tumor", IF_raw, IF), 1, true, false);
		}
		*/
		
		// Step 3.5: Crop away muscle stuff
		crop_images(dir_meas);

		// Step 4: Measure
		t0 = getTime();
		d_hypoxia = measure_hypoxic_distance(Mask_Vital, IF, 1, 2, 2);
		NF = measure_necrotic_fraction(Mask_Tumor, Mask_Vital);
		VF = measure_vascular_fraction(Mask_Vital, IF, 1);
		HF = measure_hypoxic_fraction(Mask_Vital, IF, 2);
		PF = measure_perfused_fraction(Mask_Vital, IF, 1, 3);
		r_vessel = measure_vessel_radius(Mask_Vital, IF, 1, 2);
		//perfusion_range = measure_perf_range(IF);
		t1 = getTime();
		
		print("    ---> Finished measurements in " + d2s((t1-t0)/1000, 1) + "s.");

		print("[Measurements]",	dir_sample + samples[j] +"\t" + 
					NF +"\t"+ VF +"\t" + PF +"\t"+
					HF + "\t" + r_vessel + "\t" +
					d_hypoxia);
		/*
		if(inspect){
			print("[Measurements]",	dir_sample + samples[j] +"\t" + 
											NF +"\t"+ VF +"\t" + PF +"\t"+
											HF + "\t" + r_vessel + "\t" +
											d_hypoxia + "\t" + opinion[0] + "\t" + opinion[1]);
		} else {
			print("[Measurements]",	dir_sample + samples[j] +"\t" + 
								NF +"\t"+ VF +"\t" + PF +"\t"+
								HF + "\t" + r_vessel + "\t" +
								d_hypoxia);
		}
		*/
		prettyOutputPicture(IF, Mask_Tumor, Mask_Vital, dir_meas);
		close("*");
		Ext.CLIJ2_clear();
		
	}
}

function measure_perf_range(IF_image){

	stepsize = 10;
	N_steps = 20;
	
	Ext.CLIJ2_clear();

	// Get Vascular and Hypoxic map
	Ext.CLIJ2_push(IF_image);
	Ext.CLIJ2_copySlice(IF_image, Vasc, 0);
	Ext.CLIJ2_copySlice(IF_image, Pimo, 1);
	Ext.CLIJ2_release(IF_image);

	Ext.CLIJ2_copy(Vasc, Vasc_previous);
	RangeScore = newArray();

	for (i = 0; i < N_steps; i++) {
		for (j = 0; j < 10; j++) {
			Ext.CLIJ2_dilateBox(Vasc, tmp);
			Ext.CLIJ2_dilateBox(tmp, Vasc);
		}

		Ext.CLIJ2_subtractImages(Vasc, Vasc_previous, Vasc_current);
		Ext.CLIJ2_getMeanOfMaskedPixels(Pimo, Vasc_current, mean_hypoxia);
		Ext.CLIJ2_getSumOfAllPixels(Vasc_current, Area_current);
		RangeScore = Array.concat(RangeScore, mean_hypoxia*Area_current);
		Ext.CLIJ2_copy(Vasc, Vasc_previous);
		//Ext.CLIJ2_pull(Vasc_current);
	}
	Array.show(RangeScore);
	exit();
	

	// 
}

function procIF(IF_image){
	//Function to processed a transformed, segmented IF image.	
	Ext.CLIJ2_clear();

	// convert datatype of Image
	Ext.CLIJ2_push(IF_image);
	Ext.CLIJ2_getDimensions(IF_image, width, height, depth);
	close(IF_image);
	Ext.CLIJ2_getMaximumOfAllPixels(IF_image, Max);
	Ext.CLIJ2_subtractImageFromScalar(IF_image, IF_image_conv, Max);  // seems like segmented image is signed 16-bit
	Ext.CLIJ2_convertUInt8(IF_image_conv, IF_image_uint8);
	
	Ext.CLIJ2_release(IF_image);
	Ext.CLIJ2_release(IF_image_conv);

	// Create Masks for labels
	Ext.CLIJ2_labelToMask(IF_image_uint8, IF_image_Hoechst, 3);
	Ext.CLIJ2_labelToMask(IF_image_uint8, IF_image_Vasc, 2);
	Ext.CLIJ2_labelToMask(IF_image_uint8, IF_image_Pimo, 1);
	Ext.CLIJ2_release(IF_image_uint8);

	// Post-proc Hoechst channel
	Ext.CLIJ2_multiplyImageAndScalar(IF_image_Hoechst, IF_image_Hoechst_255, 255);
	//Ext.CLIJ2_convertFloat(IF_image_Hoechst, IF_image_Hoechst_float);
	Ext.CLIJ2_gaussianBlur2D(IF_image_Hoechst_255, IF_image_Hoechst_blurred, 75, 75);
	Ext.CLIJ2_threshold(IF_image_Hoechst_blurred, IF_image_Hoechst, 0.1*255);
	//Ext.CLIJ2_release(IF_image_Hoechst_float);
	//Ext.CLIJ2_pull(IF_image_Hoechst);

	// Vital imprint
	Ext.CLIJ2_push("Mask_Vital");

	// Vasc
	Ext.CLIJ2_multiplyImages(IF_image_Vasc, "Mask_Vital", out);
	Ext.CLIJ2_pull(out);
	rename("img1");

	// Pimo
	Ext.CLIJ2_multiplyImages(IF_image_Pimo, "Mask_Vital", out);
	Ext.CLIJ2_pull(out);
	rename("img2");

	// Hoechst
	Ext.CLIJ2_multiplyImages(IF_image_Hoechst, "Mask_Vital", out);
	Ext.CLIJ2_pull(out);
	rename("img3");

	// merge and clean up
	run("Merge Channels...", "c1=img1 c2=img2 c3=img3 create");
	IF = getTitle();
	Ext.CLIJ2_clear();
	
	return IF;
}

function prettyOutputPicture(IF, TumorMask, VitalMask, dest){

	Ext.CLIJ2_clear();
	Ext.CLIJ2_push(TumorMask);
	Ext.CLIJ2_push(VitalMask);
	Ext.CLIJ2_binaryXOr(TumorMask, VitalMask, Necrosis);
	Ext.CLIJ2_pull(Necrosis);

	Necrosis = getTitle();
	setThreshold(1, 255);
	run("Create Selection");
	selectWindow(IF);
	run("Restore Selection");
	run("RGB Color");
	close(IF);
	IF = getTitle();
	
	run("Restore Selection");
	run("Set...", "value=50");
	run("Select None");
	getDimensions(width, height, channels, slices, frames);
	run("downsample ", "width=5000 height="+5000*height/width+" source=0.50 target=0.50");
	saveAs("Jpeg", dest + "Pretty_Picture.jpg");
	Ext.CLIJ2_clear();
	
}

function VitalCutOff(DL_input, IF){

	// IF image is only needed for dimensions
	Ext.CLIJ2_clear();
	selectWindow(IF);
	getDimensions(width, height, channels, slices, frames);

	Ext.CLIJ2_clear();
	selectWindow(DL_input);
	Ext.CLIJ2_push(DL_input);

	// Compress Necrosis and Vital probability maps
	close(DL_input);
	Ext.CLIJ2_argMaximumZProjection(DL_input, MaxMap, LabelMap);
	Ext.CLIJ2_release(DL_input);
	Ext.CLIJ2_release(MaxMap);

	// Scale up
	Ext.CLIJ2_pull(LabelMap);
	LabelMap1 = getTitle();
	run("Scale...", "x=- y=- width="+width+" height="+height+" interpolation=None create");
	LabelMap = getTitle();

	// Convert to masks
	Ext.CLIJ2_push(LabelMap);
	close(LabelMap1);
	close(LabelMap);
	
	Ext.CLIJ2_labelToMask(LabelMap, TumorMask, 0);
	Ext.CLIJ2_labelToMask(LabelMap, VitalMask, 2);
	Ext.CLIJ2_release(LabelMap);

	// Extract vitalmask
	Ext.CLIJ2_convertUInt8(VitalMask, VitalMask_uint8);
	Ext.CLIJ2_pull(VitalMask_uint8);
	Ext.CLIJ2_release(VitalMask);
	Ext.CLIJ2_release(VitalMask_uint8);
	rename("Mask_Vital");
	
	// Extract TumorMask
	Ext.CLIJ2_convertUInt8(TumorMask, TumorMask_uint8);
	Ext.CLIJ2_binaryNot(TumorMask_uint8, TumorMask_inv);
	Ext.CLIJ2_release(TumorMask);

	Ext.CLIJx_morphoLibJFillHoles(TumorMask_inv, tmp);
	Ext.CLIJx_morphoLibJKeepLargestRegion(tmp, TumorMask);
	Ext.CLIJ2_threshold(TumorMask, TumorMask_inv, 128);

	Ext.CLIJ2_pull(TumorMask_inv);
	rename("Mask_Tumor");
	Ext.CLIJ2_clear()
}

function crop_images(meas_directory){
	/*
	 * Function to check if image has already been crop-processed
	 */

	images = getList("image.titles");
	for (i = 0; i < images.length; i++) {
		run("Select None");
	}
	
	if(File.exists(meas_directory + "Crop.zip")){
		roiManager("open", meas_directory + "Crop.zip");
		roiManager("select", roiManager("count")-1);
	}
	/*
	} else {
		roiManager("reset");
		selectWindow(HE_image);
		waitForUser("Take your time and mark regions in the HE image that are NOT tumor tissue\nCLick Ok when done.");
		roiManager("add");
		roiManager("save", meas_directory + "Crop.zip");
	}
	*/
	
	for (i = 0; i < images.length; i++) {
		selectWindow(images[i]);
		roiManager("select", roiManager("count")-1);
		run("Clear");
	}

	//close(HE_image);
}

function inspect_raw_and_seg(images, row, evaluate, ROI){

	selectWindow(images[0]);
	getDimensions(width, height, channels, slices, frames);
	r = height/width;

	// project threshold in other images
	if(ROI){
		selectWindow(images[1]);
		setThreshold(1, 255);
		run("Create Selection");
		resetThreshold;
	}

	// iterate over images and display properly
	for (i = 0; i < images.length; i++) {
		selectWindow(images[i]);

		// make sure masks are properly displayed
		if(nSlices == 1){
			setMinAndMax(0, 1);
		}
		
		setLocation(screenWidth/3 * i, row * screenHeight/2,
					r * screenWidth/3, 	r * screenHeight/2);
		if(ROI){
			run("Restore Selection");	
		}
	}

	if (evaluate) {
		Dialog.createNonBlocking("Inspect HE");
		Dialog.addNumber("What's the quality? (1-10)", 5);
		Dialog.addString("Comment", "");
		Dialog.show();
		quality = Dialog.getNumber();
		comment = Dialog.getString();

		return newArray(quality, comment);
	}
}

function measure_hypoxic_distance(Vital, IF_image, index_vasc, index_hyp, degree_smoothing){
	// measure hypoxia weighted with distance to vitaal vessel

	df_factor = 6.0;
	Ext.CLIJ2_clear();
	
	// First, smooth vessels a bit and fill
	Ext.CLIJ2_push(IF_image);
	Ext.CLIJ2_copySlice(IF_image, vasc, index_vasc - 1);
	Ext.CLIJ2_release(IF_image);

	// constructive smoothing of vessels
	Ext.CLIJ2_closingBox(vasc, tmp, Math.pow(2, degree_smoothing));
	Ext.CLIJ2_openingBox(tmp, vasc, Math.pow(2, degree_smoothing));
	Ext.CLIJ2_release(tmp);
	
	Ext.CLIJ2_downsample2D(vasc, vasc_low_res, 1/df_factor, 1/df_factor);
	Ext.CLIJ2_release(vasc);
	Ext.CLIJ2_binaryNot(vasc_low_res, vasc);
	Ext.CLIJ2_distanceMap(vasc, DTM_low_res);
	Ext.CLIJ2_release(vasc_low_res);	
	Ext.CLIJ2_downsample2D(DTM_low_res, DTM, df_factor, df_factor);
	Ext.CLIJ2_release(DTM_low_res);
	
	// weigh hypoxia with distance transform
	Ext.CLIJ2_push(IF_image);
	Ext.CLIJ2_copySlice(IF_image, hypoxia, index_hyp - 1);
	Ext.CLIJ2_release(IF_image);
	
	//Ext.CLIJ2_convertUInt16(hypoxia, hypoxia_uint16);
	Ext.CLIJ2_multiplyImages(hypoxia, DTM, distance_weighted_hypoxia);
	Ext.CLIJ2_getMeanOfMaskedPixels(distance_weighted_hypoxia, hypoxia, mean);
	
	// convert to microns
	mean = mean * PixelSize * df_factor;
	print("    ---> Measured mean hypoxic distance: " + mean + "microns");
	Ext.CLIJ2_clear();
	close(DTM);
	close(vasc);
	Ext.CLIJ2_clear();

	return mean;
}

function measure_vessel_radius(Vital, IF_image, index_vasc, degree_smoothing){
	// elliptic fit of ellipsoid to each vessel

	Ext.CLIJ2_clear();
	run("Set Measurements...", "fit redirect=None decimal=3");
	run("Clear Results");	
	roiManager("reset");

	// First, smooth vessels a bit and fill
	Ext.CLIJ2_push(IF_image);
	Ext.CLIJ2_copySlice(IF_image, vasc, index_vasc - 1);
	
	// constructive smoothing
	Ext.CLIJ2_closingBox(vasc, tmp, Math.pow(2, degree_smoothing));
	Ext.CLIJ2_openingBox(tmp, vasc, Math.pow(2, degree_smoothing));

	// Restrict to vital area
	Ext.CLIJ2_push(Vital);
	Ext.CLIJ2_mask(vasc, Vital, vasc_vital);
	Ext.CLIJ2_pull(vasc_vital);

	// Measure
	setThreshold(1, 255);
	run("Analyze Particles...", "add");
	roiManager("Measure");
	values = newArray(nResults);

	// Poor man's median
	for (i = 0; i < nResults; i++) {
		values[i] = getResult("Minor", i);
	}
	
	values = Array.sort(values);
	Median = values[floor(values.length/2)];

	// convert to microns
	Median = Median * PixelSize;
	print("    ---> Measured median vessel radius: " + Median + " microns");
	
	close(vasc_vital);
	roiManager("reset");
	run("Clear Results");
	Ext.CLIJ2_clear();
	
	return Median;	
}

function measure_perfused_fraction(Vital, IF_image, index_vasc, index_perf){
	// Measure the hypoxic fraction based on IF image

	Ext.CLIJ2_clear();
	Hoechst = "Hoechst";
	vasc = "Vascular";
	
	selectWindow(Vital);
	Ext.CLIJ2_push(Vital);

	// get vital vessels
	Ext.CLIJ2_push(IF_image);
	Ext.CLIJ2_copySlice(IF_image, vasc, index_vasc - 1);
	Ext.CLIJ2_binaryAnd(vasc, Vital, vasc_vital);
	Ext.CLIJ2_sumOfAllPixels(vasc_vital);
	Area_CD31 = getResult("Sum", nResults-1);

	// get perfused fraction
	Ext.CLIJ2_copySlice(IF_image, Hoechst, index_perf - 1);
	Ext.CLIJ2_binaryAnd(vasc_vital, Hoechst, perfused);
	Ext.CLIJ2_sumOfAllPixels(perfused);
	Area_perfused = getResult("Sum", nResults-1);

	close(Hoechst);
	close(vasc);
	Ext.CLIJ2_clear();

	print("    ---> Measured perfused fraction: " + Area_perfused/Area_CD31);

	return Area_perfused/Area_CD31;
}

function measure_necrotic_fraction(Tumor, Vital){
	Ext.CLIJ2_clear();
	Ext.CLIJ2_push(Tumor);
	Ext.CLIJ2_push(Vital);

	Ext.CLIJ2_getSumOfAllPixels(Tumor, A_Tumor);
	Ext.CLIJ2_getSumOfAllPixels(Vital, A_Vital);

	print("    ---> Measured necrotic fraction: " + 1.0 - A_Vital/A_Tumor);
	Ext.CLIJ2_clear();
	return (1.0 - A_Vital/A_Tumor);
	
}

function measure_vascular_fraction(Vital, IF_image, index){

	Ext.CLIJ2_clear();
	vasc = "Vascular";

	// measure vital
	Ext.CLIJ2_push(Vital);
	Ext.CLIJ2_sumOfAllPixels(Vital);
	Area_Vital = getResult("Sum", nResults-1);
	
	// measure intersection
	Ext.CLIJ2_push(IF_image);
	Ext.CLIJ2_copySlice(IF_image, vasc, index - 1);
	Ext.CLIJ2_binaryAnd(vasc, Vital, vasc_vital);
	Ext.CLIJ2_sumOfAllPixels(vasc_vital);
	Area_CD31 = getResult("Sum", nResults-1);
	Ext.CLIJ2_clear();
	close(vasc);

	print("    ---> Measured vascular fraction: " + Area_CD31/Area_Vital);
	return Area_CD31/Area_Vital;
	
}

function measure_hypoxic_fraction(Vital, IF_image, index){
	
	Ext.CLIJ2_clear();
	Hypoxia = "Hypoxia";

	// measure vital
	Ext.CLIJ2_push(Vital);
	Ext.CLIJ2_sumOfAllPixels(Vital);
	Area_Vital = getResult("Sum", nResults-1);
	
	// measure intersection
	Ext.CLIJ2_push(IF_image);
	Ext.CLIJ2_copySlice(IF_image, Hypoxia, index - 1);
	Ext.CLIJ2_binaryAnd(Hypoxia, Vital, Hypoxia_vital);
	Ext.CLIJ2_sumOfAllPixels(Hypoxia_vital);
	Area_Pimo = getResult("Sum", nResults-1);
	Ext.CLIJ2_clear();
	close(Hypoxia);

	print("    ---> Measured hypoxic fraction: " + Area_Pimo/Area_Vital);
	return Area_Pimo/Area_Vital;
	
	
}

function getsubdirs(path){
 	// Returns array with all subdirectories in <path>
 	list = getFileList(path);
 	dirlist = newArray();
 	for (i = 0; i < list.length; i++) {
 		if (File.isDirectory(path + list[i])) {
 			dirlist = Array.concat(dirlist, list[i]);
 		}
 	}
 	return dirlist;
}

function findFileByID(path, ID){
	// browse directory <path> and check for files that match the given ID string <ID>
	files = newArray();
	filelist = getFileList(path);
	for (i = 0; i < filelist.length; i++) {

		// exclude ROIs from search
		if (endsWith(filelist[i], "roi")) {
			continue;
		}
		
		if (matches(filelist[i], ".*" + ID + ".*")) {
			files = Array.concat(files, filelist[i]);
		}
	}

	print("    Found " + d2s(files.length, 0) +" files matching " + ID + " in " + path);
	return files;
}
