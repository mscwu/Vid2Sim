using UnityEngine;

[RequireComponent(typeof(Camera))]
public class DepthCaptureEffect : MonoBehaviour
{
    public Material depthMaterial;
    public RenderTexture depthTexture;

    private void Start()
    {
        if (depthTexture == null)
        {
            depthTexture = new RenderTexture(Screen.width, Screen.height, 0, RenderTextureFormat.ARGBFloat);
        }
    }

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth;
        Graphics.Blit(source, depthTexture, depthMaterial);
        Graphics.Blit(depthTexture, destination);
    }
}