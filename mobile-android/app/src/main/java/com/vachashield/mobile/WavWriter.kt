package com.vachashield.mobile

import java.io.BufferedOutputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream

object WavWriter {
    fun writePcm16MonoWav(file: File, pcmData: ByteArray, sampleRate: Int) {
        val channels = 1
        val bitsPerSample = 16
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8
        val dataChunkSize = pcmData.size
        val riffChunkSize = 36 + dataChunkSize

        DataOutputStream(BufferedOutputStream(FileOutputStream(file))).use { out ->
            out.writeAscii("RIFF")
            out.writeIntLE(riffChunkSize)
            out.writeAscii("WAVE")

            out.writeAscii("fmt ")
            out.writeIntLE(16)
            out.writeShortLE(1)
            out.writeShortLE(channels.toShort())
            out.writeIntLE(sampleRate)
            out.writeIntLE(byteRate)
            out.writeShortLE(blockAlign.toShort())
            out.writeShortLE(bitsPerSample.toShort())

            out.writeAscii("data")
            out.writeIntLE(dataChunkSize)
            out.write(pcmData)
        }
    }

    private fun DataOutputStream.writeAscii(value: String) {
        writeBytes(value)
    }

    private fun DataOutputStream.writeIntLE(value: Int) {
        write(value and 0xFF)
        write(value shr 8 and 0xFF)
        write(value shr 16 and 0xFF)
        write(value shr 24 and 0xFF)
    }

    private fun DataOutputStream.writeShortLE(value: Short) {
        val intValue = value.toInt()
        write(intValue and 0xFF)
        write(intValue shr 8 and 0xFF)
    }
}
