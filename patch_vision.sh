#!/bin/bash
cat src/vision.cpp | grep -n "static bool g_tracking_active"
