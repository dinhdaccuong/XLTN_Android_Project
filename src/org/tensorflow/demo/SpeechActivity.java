/*
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Demonstrates how to run an audio recognition model in Android.

This example loads a simple speech recognition model trained by the tutorial at
https://www.tensorflow.org/tutorials/audio_training

The model files should be downloaded automatically from the TensorFlow website,
but if you have a custom model you can update the LABEL_FILENAME and
MODEL_FILENAME constants to point to your own files.

The example application displays a list view with all of the known audio labels,
and highlights each one when it thinks it has detected one through the
microphone. The averaging of results to give a more reliable signal happens in
the RecognizeCommands helper class.
*/

package org.tensorflow.demo;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothManager;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ListView;
import android.widget.Toast;

import com.google.android.gms.appindexing.Action;
import com.google.android.gms.appindexing.AppIndex;
import com.google.android.gms.common.api.GoogleApiClient;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.locks.ReentrantLock;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/**
 * An activity that listens for audio and then uses a TensorFlow model to detect particular classes,
 * by default a small set of action words.
 */
public class SpeechActivity extends Activity {

    // UI
    ListView lvDeviceScaned;
    Button buttonConnect;
    Button buttonScan;
    Button buttonVoice;
    Button buttonLedControl;
    Button buttonMotorControl;

    // BLE
    SimpleDeviceInfoAdapter adapterDeviceInfo;
    ArrayList<SimpleDeviceInfo> listDeviceInfo;
    ArrayList<BluetoothDevice> listDeviceDiscovered;
    Set<SimpleDeviceInfo> setDeviceInfo;

    BluetoothManager bleManager;
    BluetoothAdapter bleAdapter;
    BluetoothLeScanner bleScanner;
    BluetoothGatt bleGatt;
    BluetoothGattService myGattService = null;
    BluetoothGattCharacteristic myCharacteristic = null;
    BluetoothGattDescriptor myDescriptor = null;
    ScanSettings settings;
    boolean isScanning = false;
    boolean isConnected = false;

    int ledStatus = 0; // led off
    int motorStatus = 0; // motor stop, 1 motor turn left, 2 motor turn right

    private static final int PERMISSION_REQUEST_COARSE_LOCATION = 1;
    private static UUID SERVICE_UUID = UUID.fromString("feed0001-c497-4476-a7ed-727de7648ab1");
    private Handler mHandler = new Handler();

    // Server
    String myServiceUUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
    private static final long SCAN_PERIOD = 15000;
    private GoogleApiClient client;
    String log_tag = "BLE_CUONG";

    // Tensorflow
    // Constants that control the behavior of the recognition code and model
    // settings. See the audio recognition tutorial for a detailed explanation of
    // all these, but you should customize them to match your training settings if
    // you are running your own model.
    private static final int SAMPLE_RATE = 16000;
    private static final int SAMPLE_DURATION_MS = 1000;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
    private static final long AVERAGE_WINDOW_DURATION_MS = 500;
    private static final float DETECTION_THRESHOLD = 0.70f;
    private static final int SUPPRESSION_MS = 1500;
    private static final int MINIMUM_COUNT = 3;
    private static final long MINIMUM_TIME_BETWEEN_SAMPLES_MS = 30;
    private static final String LABEL_FILENAME = "file:///android_asset/conv_actions_labels.txt";
    private static final String MODEL_FILENAME = "file:///android_asset/conv_actions_frozen.pb";
    private static final String INPUT_DATA_NAME = "decoded_sample_data:0";
    private static final String SAMPLE_RATE_NAME = "decoded_sample_data:1";
    private static final String OUTPUT_SCORES_NAME = "labels_softmax";


    // UI elements.
    private static final int REQUEST_RECORD_AUDIO = 13;
    private static final String LOG_TAG = SpeechActivity.class.getSimpleName();

    // Working variables.
    short[] recordingBuffer = new short[RECORDING_LENGTH];
    int recordingOffset = 0;
    boolean shouldContinue = true;
    private Thread recordingThread;
    boolean shouldContinueRecognition = true;
    private Thread recognitionThread;
    private final ReentrantLock recordingBufferLock = new ReentrantLock();
    private TensorFlowInferenceInterface inferenceInterface;
    private List<String> labels = new ArrayList<String>();
    private List<String> displayedLabels = new ArrayList<>();
    private RecognizeCommands recognizeCommands = null;

    private boolean isRecogniting = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Set up the UI.
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_speech);

        InitView();
        Log.d(log_tag, "Hereeeee");
        // Doc file text de hien len labels list
        String actualFilename = LABEL_FILENAME.split("file:///android_asset/")[1];  // Tach chuoi = conv_actions_lbels.txt

        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(getAssets().open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!", e);
        }
        // Set up an object to smooth recognition results to increase accuracy.
        recognizeCommands =
                new RecognizeCommands(
                        labels,
                        AVERAGE_WINDOW_DURATION_MS,
                        DETECTION_THRESHOLD,
                        SUPPRESSION_MS,
                        MINIMUM_COUNT,
                        MINIMUM_TIME_BETWEEN_SAMPLES_MS);

        // Load the TensorFlow model.
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILENAME);

        // Start the recording and recognition threads.
        requestMicrophonePermission();

        // BLE
        // Make sure we have access coarse location enabled, if not, prompt the user to enable it
        if (this.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            final AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setTitle("This app needs location access");
            builder.setMessage("Please grant location access so this app can detect peripherals.");
            builder.setPositiveButton(android.R.string.ok, null);
            builder.setOnDismissListener(new DialogInterface.OnDismissListener() {
                @Override
                public void onDismiss(DialogInterface dialog) {
                    requestPermissions(new String[]{Manifest.permission.ACCESS_COARSE_LOCATION}, PERMISSION_REQUEST_COARSE_LOCATION);
                }
            });
            builder.show();
        }
        client = new GoogleApiClient.Builder(this).addApi(AppIndex.API).build();
        // Init BLE
        bleManager = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
        bleAdapter = bleManager.getAdapter();

        if (bleAdapter == null || !bleAdapter.isEnabled()) {
            Log.d(log_tag, "BLE adapter is not enable!");
            return;
        } else {
            Log.d(log_tag, "BLE adapter init successfully!");
        }

        bleScanner = bleAdapter.getBluetoothLeScanner();
        settings = new ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).setReportDelay(500).build();
    }

// end onCreate
    private void InitView(){
        listDeviceInfo = new ArrayList<SimpleDeviceInfo>();
        adapterDeviceInfo = new SimpleDeviceInfoAdapter(this, R.layout.devices_scaned, listDeviceInfo);
        setDeviceInfo = new LinkedHashSet<SimpleDeviceInfo>();
        listDeviceDiscovered = new ArrayList<BluetoothDevice>();

        lvDeviceScaned = (ListView) findViewById(R.id.listViewDevives);
        lvDeviceScaned.setAdapter(adapterDeviceInfo);
        lvDeviceScaned.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
                view.setSelected(true);
                if (isScanning) {
                    isScanning = false;
                    buttonScan.setBackgroundColor(getResources().getColor(R.color.button_release));
                    buttonScan.setText("Start Scanning");
                    mHandler.removeCallbacksAndMessages(null);
                    stopScanning();
                }
                lvDeviceScaned.setTag(listDeviceInfo.get(i));
            }
        });

        buttonLedControl = findViewById(R.id.buttonLed);
        buttonLedControl.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(!isConnected || myCharacteristic == null)
                    return;
                if(ledStatus == 0) {   // if led is off
                    SendCommandToDevice(CommandToWrite.COMMAND_TURN_ON_LED);
                }
                else {
                    SendCommandToDevice(CommandToWrite.COMMAND_TURN_OFF_LED);
                }
            }
        });
        buttonMotorControl = findViewById(R.id.buttonMotor);
        buttonMotorControl.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(!isConnected || myCharacteristic == null)
                    return;
                switch (motorStatus){
                    case 0:
                        SendCommandToDevice(CommandToWrite.COMMAND_GO_MOTOR);
                        break;
                    case 1:
                        SendCommandToDevice(CommandToWrite.COMMAND_RIGHT_MOTOR);
                        break;
                    case 2:
                        SendCommandToDevice(CommandToWrite.COMMAND_STOP_MOTOR);
                        break;
                }
            }
        });

        buttonScan = findViewById(R.id.buttonScan);
        buttonScan.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(!isScanning){
                    isScanning = true;
                    clearListView();
                    buttonScan.setBackgroundColor(getResources().getColor(R.color.button_push));
                    buttonScan.setText("Stop Scanning");
                    startScanning();
                }
                else{
                    isScanning = false;
                    buttonScan.setBackgroundColor(getResources().getColor(R.color.button_release));
                    buttonScan.setText("Start Scanning");
                    mHandler.removeCallbacksAndMessages(null);
                    stopScanning();
                }
            }
        });

        buttonConnect = findViewById(R.id.buttonConnect);
        buttonConnect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(!isConnected){
                    connectToDeviceSelected();
                }
                else{
                    disconnectedDevice();
                }
            }
        });


        buttonVoice = findViewById(R.id.buttonVoice);
        buttonVoice.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isRecogniting) {
                    isRecogniting = true;
                    buttonVoice.setBackgroundColor(getResources().getColor(R.color.button_push));
                    startRecording();
                    startRecognition();
                } else {
                    isRecogniting = false;
                    buttonVoice.setBackgroundColor(getResources().getColor(R.color.button_release));
                    stopRecording();
                    stopRecognition();
                }
            }
        });
    }
    private void startScanning() {
        Log.d(log_tag, "Start scanning!");
        AsyncTask.execute(new Runnable() {
            @Override
            public void run() {
                bleScanner.startScan(null, settings, scanCallback);
            }
        });

        mHandler.postDelayed(new Runnable() {
            @Override
            public void run() {
                stopScanning();
            }
        }, SCAN_PERIOD);
    }

    private void stopScanning() {
        Log.d(log_tag, "Stop scanning!");
        AsyncTask.execute(new Runnable() {
            @Override
            public void run() {
                bleScanner.stopScan(scanCallback);
            }
        });
    }
    private void clearListView() {
        listDeviceInfo.clear();
        adapterDeviceInfo.notifyDataSetChanged();
    }

    private void connectToDeviceSelected(){
        Log.d(log_tag, "Trying to connect...");
        Object ob = lvDeviceScaned.getTag();
        if(ob == null)
            return;
        SimpleDeviceInfo spdvInfor = (SimpleDeviceInfo)ob;
        bleGatt = listDeviceDiscovered.get(spdvInfor.getDeviceIndex()).connectGatt(this, false, bleGattCallback);
    }

    private void disconnectedDevice(){
        Log.d(log_tag, "Trying to disconnect...");
        bleGatt.disconnect();
    }
    // Device scan callback
    private final ScanCallback scanCallback = new ScanCallback() {
        @Override
        public void onScanResult(int callbackType, ScanResult result) {
            super.onScanResult(callbackType, result);
            //Log.d(log_tag, result.getDevice().getName());
            Log.d(log_tag, "Search device!");
        }

        @Override
        public void onBatchScanResults(List<ScanResult> results) {
            super.onBatchScanResults(results);
            if (!results.isEmpty()) {
                setDeviceInfo.clear();  // Xóa set
                for (int i = 0; i < results.size(); i++) {      // Lặp lại địa chỉ mac
                    ScanResult result = results.get(i);
                    SimpleDeviceInfo spDeviceInfo = new SimpleDeviceInfo(i, result.getDevice().getName() + "", result.getDevice().getAddress(), "rssi: " + result.getRssi());
                    setDeviceInfo.add(spDeviceInfo);    // Thêm vào set
                }

                listDeviceInfo.clear();
                listDeviceDiscovered.clear();   // Xóa list cũ
                for (SimpleDeviceInfo dv : setDeviceInfo) {
                    listDeviceInfo.add(dv);
                    listDeviceDiscovered.add(results.get(dv.getDeviceIndex()).getDevice()); // indexDevice trong device infor cũn là chỉ số trong results
                }
                adapterDeviceInfo.notifyDataSetChanged();
            }
        }

        @Override
        public void onScanFailed(int errorCode) {
            super.onScanFailed(errorCode);
            Log.d(log_tag, "Scan failed");
        }
    };

    private void SendCommandToDevice(byte command){
        if(!isConnected && myCharacteristic == null)
            return;
        byte[] byteToWrite = {command};
        myCharacteristic.setValue(byteToWrite);
        bleGatt.writeCharacteristic(myCharacteristic);
    }

    private void ExecuteResponseCommand(byte command)
    {
        switch (command)
        {
            case 0:     // led turned On
                ledStatus = 1;
                buttonLedControl.setBackgroundColor(getResources().getColor(R.color.led_on));
                buttonLedControl.setText("LED ON");
                Log.d(log_tag, "Led turned on");
                break;
            case 1:     // led turned Off
                ledStatus = 0;
                buttonLedControl.setBackgroundColor(getResources().getColor(R.color.led_off));
                buttonLedControl.setText("LED OFF");
                Log.d(log_tag, "Led turned off");
                break;
            case 2:     // Motor go
                motorStatus = 1;
                buttonMotorControl.setBackgroundColor(getResources().getColor(R.color.motor_turnLeft));
                buttonMotorControl.setText("Motor turn left");
                break;
            case 3:     // Motor left
                motorStatus = 1;
                buttonMotorControl.setBackgroundColor(getResources().getColor(R.color.motor_turnLeft));
                buttonMotorControl.setText("Motor turn left");
                break;
            case 4:     // Motor right
                motorStatus = 2;
                buttonMotorControl.setBackgroundColor(getResources().getColor(R.color.motor_turnRight));
                buttonMotorControl.setText("Motor turn right");
                break;
            case 5:     // Motor stop
                motorStatus = 0;
                buttonMotorControl.setBackgroundColor(getResources().getColor(R.color.motor_stop));
                buttonMotorControl.setText("Motor stop");
                break;
            default:
                break;
        }
    }

    private final BluetoothGattCallback bleGattCallback = new BluetoothGattCallback() {
        @Override
        public void onConnectionStateChange(BluetoothGatt gatt, int status, int newState) {
            super.onConnectionStateChange(gatt, status, newState);
            Log.d(log_tag, "onConnectionStateChange");
            switch (newState){
                case 0:         // device disconnected
                    isConnected = false;
                    buttonConnect.setBackgroundColor(getResources().getColor(R.color.button_release));
                    buttonConnect.setText("Connect");
                    Log.d(log_tag, "onConnectionStateChange: Device disconnected");
                    break;
                case 2:         // device connected
                    isConnected = true;
                    buttonConnect.setBackgroundColor(getResources().getColor(R.color.button_push));
                    buttonConnect.setText("Disconnect");
                    Log.d(log_tag, "onConnectionStateChange: Device connected");
                    bleGatt.discoverServices();

                    break;
                default:        // another state
                    Log.d(log_tag, "onConnectionStateChange: Another state");
                    break;
            }
        }

        @Override
        public void onServicesDiscovered(BluetoothGatt gatt, int status) {
            super.onServicesDiscovered(gatt, status);

            List<BluetoothGattService> listBLEGattService = bleGatt.getServices();
            if(listBLEGattService == null){
                Log.d(log_tag, "Could not find any services!");
                return;
            }
            Log.d(log_tag, "onServicesDiscovered");
            for(BluetoothGattService bleGatt: listBLEGattService){
                String sUUID = bleGatt.getUuid().toString();
                if(sUUID.equals(myServiceUUID)){
                    myGattService = bleGatt;
                    break;
                }
            }
            if(myGattService == null)
            {
                Log.d(log_tag, "Cound not find the Your GatService!");
                return;
            }
            Log.d(log_tag, "Found Your GattService!");
            List<BluetoothGattCharacteristic> listCharacteristic = myGattService.getCharacteristics();
            if(!listCharacteristic.isEmpty()){
                myCharacteristic = listCharacteristic.get(0);
                Log.d(log_tag, "My characteristic: " + myCharacteristic.getUuid().toString());
            }

            if(myCharacteristic == null)
                return;

            List<BluetoothGattDescriptor> listDescriptor = myCharacteristic.getDescriptors();
            if(!listDescriptor.isEmpty()){
                myDescriptor = listDescriptor.get(0);
                Log.d(log_tag, "My Descriptor: " + myDescriptor.getUuid().toString());
            }

            // Enable notify
            bleGatt.setCharacteristicNotification(myCharacteristic, true);
            myDescriptor.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
            bleGatt.writeDescriptor(myDescriptor);
        }

        @Override
        public void onCharacteristicRead(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic, int status) {
            super.onCharacteristicRead(gatt, characteristic, status);
            Log.d(log_tag, "onCharacteristicRead");
        }

        @Override
        public void onCharacteristicWrite(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic, int status) { // ESP32 write
            super.onCharacteristicWrite(gatt, characteristic, status);
            //Log.d(log_tag, "onCharacteristicWrite");
        }

        @Override
        public void onCharacteristicChanged(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic) {
            super.onCharacteristicChanged(gatt, characteristic);
            //Log.d(log_tag, "onCharacteristicChanged");
            byte[] byteToRead = myCharacteristic.getValue();
            if(byteToRead.length <= 0)
                return;
            byte byte0 = byteToRead[0];
            ExecuteResponseCommand(byte0);
        }
    };

    @Override
    public void onStart() {
        super.onStart();

        client.connect();
        Action viewAction = Action.newAction(
                Action.TYPE_VIEW, // TODO: choose an action type.
                "Main Page", // TODO: Define a title for the content shown.
                // TODO: If you have web page content that matches this app activity's content,
                // make sure this auto-generated web page URL is correct.
                // Otherwise, set the URL to null.
                Uri.parse("http://host/path"),
                // TODO: Make sure this auto-generated app URL is correct.
                Uri.parse("android-app://org.tensorflow.demo/http/host/path")
        );
        AppIndex.AppIndexApi.start(client, viewAction);
    }

    @Override
    public void onStop() {
        super.onStop();

        Action viewAction = Action.newAction(
                Action.TYPE_VIEW, // TODO: choose an action type.
                "Main Page", // TODO: Define a title for the content shown.
                // TODO: If you have web page content that matches this app activity's content,
                // make sure this auto-generated web page URL is correct.
                // Otherwise, set the URL to null.
                Uri.parse("http://host/path"),
                // TODO: Make sure this auto-generated app URL is correct.
                Uri.parse("android-app://org.tensorflow.demo/http/host/path")
        );
        AppIndex.AppIndexApi.end(client, viewAction);
        client.disconnect();
    }

    ///////////////////////////////////////////////////////////////////////////////////
    private void requestMicrophonePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(
                    new String[]{android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQUEST_RECORD_AUDIO
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Log.d(log_tag, "onRequestPermissionsResult");
            //startRecording();
            //startRecognition();
        }
    }

    public synchronized void startRecording() {
        if (recordingThread != null) {
            return;
        }
        shouldContinue = true;
        recordingThread =
                new Thread(
                        new Runnable() {
                            @Override
                            public void run() {
                                record();
                            }
                        });
        recordingThread.start();
    }

    public synchronized void stopRecording() {
        if (recordingThread == null) {
            return;
        }
        shouldContinue = false;
        recordingThread = null;
    }

    private void record() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);
        // Estimate the buffer size we'll need for this device.
        int bufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * 2;
        }
        short[] audioBuffer = new short[bufferSize / 2];

        AudioRecord record =
                new AudioRecord(
                        MediaRecorder.AudioSource.DEFAULT,
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        bufferSize);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(LOG_TAG, "Audio Record can't initialize!");
            return;
        }

        record.startRecording();

        Log.v(LOG_TAG, "Start recording");

        // Loop, gathering audio data and copying it to a round-robin buffer.
        while (shouldContinue) {
            int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
            int maxLength = recordingBuffer.length;
            int newRecordingOffset = recordingOffset + numberRead;
            int secondCopyLength = Math.max(0, newRecordingOffset - maxLength);
            int firstCopyLength = numberRead - secondCopyLength;
            // We store off all the data for the recognition thread to access. The ML
            // thread will copy out of this buffer into its own, while holding the
            // lock, so this should be thread safe.
            recordingBufferLock.lock();
            try {
                System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, firstCopyLength);
                System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer, 0, secondCopyLength);
                recordingOffset = newRecordingOffset % maxLength;
            } finally {
                recordingBufferLock.unlock();
            }
        }

        record.stop();
        record.release();
    }

    // end record
    public synchronized void startRecognition() {
        if (recognitionThread != null) {
            return;
        }
        shouldContinueRecognition = true;
        recognitionThread =
                new Thread(
                        new Runnable() {
                            @Override
                            public void run() {
                                recognize();
                            }
                        });
        recognitionThread.start();
    }

    public synchronized void stopRecognition() {
        if (recognitionThread == null) {
            return;
        }
        shouldContinueRecognition = false;
        recognitionThread = null;
    }

    private void recognize() {
        Log.v(LOG_TAG, "Start recognition");

        short[] inputBuffer = new short[RECORDING_LENGTH];
        float[] floatInputBuffer = new float[RECORDING_LENGTH];
        float[] outputScores = new float[labels.size()];
        String[] outputScoresNames = new String[]{OUTPUT_SCORES_NAME};
        int[] sampleRateList = new int[]{SAMPLE_RATE};

        // Loop, grabbing recorded data and running the recognition model on it.
        while (shouldContinueRecognition) {
            // The recording thread places data in this round-robin buffer, so lock to
            // make sure there's no writing happening and then copy it to our own
            // local version.
            recordingBufferLock.lock();
            try {
                int maxLength = recordingBuffer.length;
                int firstCopyLength = maxLength - recordingOffset;
                int secondCopyLength = recordingOffset;
                System.arraycopy(recordingBuffer, recordingOffset, inputBuffer, 0, firstCopyLength);
                System.arraycopy(recordingBuffer, 0, inputBuffer, firstCopyLength, secondCopyLength);
            } finally {
                recordingBufferLock.unlock();
            }

            // We need to feed in float values between -1.0f and 1.0f, so divide the
            // signed 16-bit inputs.
            for (int i = 0; i < RECORDING_LENGTH; ++i) {
                floatInputBuffer[i] = inputBuffer[i] / 32767.0f;
            }

            // Run the model.
            inferenceInterface.feed(SAMPLE_RATE_NAME, sampleRateList);
            inferenceInterface.feed(INPUT_DATA_NAME, floatInputBuffer, RECORDING_LENGTH, 1);
            inferenceInterface.run(outputScoresNames);
            inferenceInterface.fetch(OUTPUT_SCORES_NAME, outputScores);

            // Use the smoother to figure out if we've had a real recognition event.
            long currentTime = System.currentTimeMillis();
            final RecognizeCommands.RecognitionResult result =
                    recognizeCommands.processLatestResults(outputScores, currentTime);

            runOnUiThread(
                    new Runnable() {
                        @Override
                        public void run() {
                            // If we do have a new command, highlight the right list entry.
                            if (!result.foundCommand.startsWith("_") && result.isNewCommand) {
                                int labelIndex = -1;
                                for (int i = 0; i < labels.size(); ++i) {
                                    if (labels.get(i).equals(result.foundCommand)) {
                                        labelIndex = i;
                                    }
                                }
                                switch (labels.get(labelIndex)){
                                    case "on":
                                        SendCommandToDevice(CommandToWrite.COMMAND_TURN_ON_LED);
                                        break;
                                    case "off":
                                        SendCommandToDevice(CommandToWrite.COMMAND_TURN_OFF_LED);
                                        break;
                                    case "left":
                                        SendCommandToDevice(CommandToWrite.COMMAND_LEFT_MOTOR);
                                        break;
                                    case "right":
                                        SendCommandToDevice(CommandToWrite.COMMAND_RIGHT_MOTOR);
                                        break;
                                    case "stop":
                                        SendCommandToDevice(CommandToWrite.COMMAND_STOP_MOTOR);
                                        break;
                                    case "go":
                                        SendCommandToDevice(CommandToWrite.COMMAND_GO_MOTOR);
                                        break;
                                }
                                //Toast.makeText(SpeechActivity.this, labels.get(labelIndex), Toast.LENGTH_SHORT).show();
                            }
                        }
                    });
            try {
                // We don't need to run too frequently, so snooze for a bit.
                Thread.sleep(MINIMUM_TIME_BETWEEN_SAMPLES_MS);
            } catch (InterruptedException e) {
                // Ignore
            }
        }

        Log.v(LOG_TAG, "End recognition");
    }

    // BLE

}
