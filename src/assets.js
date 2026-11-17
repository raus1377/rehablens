import { AssetType, defineAssets } from '@iwsdk/core';
import { createStudioFloor } from './scene-assets/studio.scene-asset.js';
const publicAssetUrl = (filePath) => `${import.meta.env.BASE_URL}${filePath.replace(/^\/+/u, '')}`;
export default defineAssets({
    'studio-floor': createStudioFloor(),
    'welcome-panel': {
        url: publicAssetUrl('ui/welcome.uikitml'),
        type: AssetType.UIKitML,
        name: 'Welcome Panel',
    },
});
