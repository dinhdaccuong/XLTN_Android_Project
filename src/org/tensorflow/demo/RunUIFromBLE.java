package org.tensorflow.demo;

import android.bluetooth.le.ScanResult;

import java.util.List;

public interface RunUIFromBLE {
    public abstract void onUIStartScanning();
    public abstract void onUIStopScanning();
    public abstract void onUIConnect();
    public abstract void onUIDisconnect();
    public abstract void onUIBatchScanResults(List<ScanResult> results);
}
