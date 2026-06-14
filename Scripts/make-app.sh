#!/usr/bin/env bash
# Builds Mandelbrot.app in the project root.
set -euo pipefail

cd "$(dirname "$0")/.."

APP_NAME="Mandelbrot"
BUNDLE_ID="org.madscientistroom.Mandelbrot"
BUNDLE_VERSION="1.0"
APP_DIR="${APP_NAME}.app"
BUILD_DIR=".build/release"
ICON_WORK=".build/icon"

echo "==> Building release binaries"
swift build -c release

echo "==> Generating app icon"
rm -rf "$ICON_WORK"
mkdir -p "$ICON_WORK"
"$BUILD_DIR/IconGenerator" "$ICON_WORK/icon_1024.png"

ICONSET="$ICON_WORK/AppIcon.iconset"
mkdir -p "$ICONSET"
SRC="$ICON_WORK/icon_1024.png"

# Icon sizes per Apple's iconutil convention.
sips -z   16   16 "$SRC" --out "$ICONSET/icon_16x16.png"       >/dev/null
sips -z   32   32 "$SRC" --out "$ICONSET/icon_16x16@2x.png"    >/dev/null
sips -z   32   32 "$SRC" --out "$ICONSET/icon_32x32.png"       >/dev/null
sips -z   64   64 "$SRC" --out "$ICONSET/icon_32x32@2x.png"    >/dev/null
sips -z  128  128 "$SRC" --out "$ICONSET/icon_128x128.png"     >/dev/null
sips -z  256  256 "$SRC" --out "$ICONSET/icon_128x128@2x.png"  >/dev/null
sips -z  256  256 "$SRC" --out "$ICONSET/icon_256x256.png"     >/dev/null
sips -z  512  512 "$SRC" --out "$ICONSET/icon_256x256@2x.png"  >/dev/null
sips -z  512  512 "$SRC" --out "$ICONSET/icon_512x512.png"     >/dev/null
cp "$SRC" "$ICONSET/icon_512x512@2x.png"

iconutil -c icns "$ICONSET" -o "$ICON_WORK/AppIcon.icns"

echo "==> Assembling $APP_DIR"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

cp "$BUILD_DIR/MandelbrotApp" "$APP_DIR/Contents/MacOS/$APP_NAME"
cp "$ICON_WORK/AppIcon.icns" "$APP_DIR/Contents/Resources/AppIcon.icns"

cat > "$APP_DIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${BUNDLE_VERSION}</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.graphics-design</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
</dict>
</plist>
EOF

echo "APPL????" > "$APP_DIR/Contents/PkgInfo"

# Touch the bundle so Launch Services picks up the new icon.
touch "$APP_DIR"

echo "==> Built $APP_DIR"
