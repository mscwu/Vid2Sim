using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityGLTF;

public static class GLBLoader
{
    public static bool IsLoadingComplete { get; private set; }
    private static int filesToLoad;
    private static int filesLoaded;
    private static List<Bounds> placedObjectBounds = new List<Bounds>();

    /// <summary>
    /// Resets the IsLoadingComplete flag to false.
    /// </summary>
    public static void ResetLoadingComplete()
    {
        IsLoadingComplete = false;
    }

    /// <summary>
    /// Loads and places GLB assets deterministically using a provided System.Random instance.
    public static void LoadAndPlaceAssets(int maxObjNumber, string folderPath, GameObject parentObject, GameObject generateRegion, MonoBehaviour monoBehaviour, System.Random sysRandom, float scaleFactor = 1.2f)
    {
        DeleteAllChildren(parentObject);
        Debug.Log($"Loading assets from: {folderPath}");

        if (generateRegion == null)
        {
            Debug.LogError("GenerateRegion is not assigned.");
            return;
        }

        // Retrieve all GLB files from the specified folder
        string[] glbFiles = Directory.GetFiles(folderPath, "*.glb");

        if (glbFiles.Length == 0)
        {
            Debug.LogError("No GLB files found in the specified folder.");
            IsLoadingComplete = true; // No files to load, so loading is complete
            return;
        }

        // Determine the number of files to load using System.Random
        int N = sysRandom.Next(1, maxObjNumber + 1); // Ensure at least one object is loaded
        Debug.Log($"Attempting to load {N} GLB file(s).");

        if (N == 0)
        {
            IsLoadingComplete = true; // No files to load, so loading is complete
            return;
        }

        // Randomly select N files from the folder using sysRandom
        string[] selectedFiles = glbFiles.OrderBy(x => sysRandom.NextDouble()).Take(N).ToArray();
        filesToLoad = selectedFiles.Length;

        if (filesToLoad == 0)
        {
            Debug.LogWarning("Selected zero files to load.");
            IsLoadingComplete = true;
            return;
        }

        Debug.Log("Selected GLB files:");
        foreach (string filePath in selectedFiles)
        {
            Debug.Log($" - {filePath}");
        }

        // Start sequential loading to maintain deterministic order
        monoBehaviour.StartCoroutine(LoadGLBSequentially(selectedFiles, parentObject, generateRegion, sysRandom, scaleFactor));
    }

    /// <summary>
    /// Coroutine to load GLB files one after another.
    /// </summary>
    private static IEnumerator LoadGLBSequentially(string[] selectedFiles, GameObject parentObject, GameObject generateRegion, System.Random sysRandom, float scaleFactor)
    {
        foreach (string filePath in selectedFiles)
        {
            Vector3 position = GetRandomPosition(generateRegion, sysRandom);
            yield return LoadGLB(filePath, parentObject, generateRegion, position, scaleFactor);
        }
        IsLoadingComplete = true;
        Debug.Log("All GLB files have been loaded.");
    }

    /// <summary>
    /// Coroutine to load a single GLB file and place it in the scene.
    /// </summary>
    private static IEnumerator LoadGLB(string filePath, GameObject parentObject, GameObject generateRegion, Vector3 position, float scaleFactor)
    {
        string url = new System.Uri(filePath).AbsoluteUri;
        Debug.Log($"Loading GLB from URL: {url}");

        GLTFSceneImporter importer = new GLTFSceneImporter(
            url,
            new ImportOptions()
        );

        bool loadFailed = false;

        // Load the GLB scene asynchronously
        yield return importer.LoadSceneAsync(
            sceneIndex: -1,
            onLoadComplete: (gltfRoot, loadScenes) =>
            {
                if (gltfRoot == null)
                {
                    loadFailed = true;
                    Debug.LogError($"Failed to load GLB file: {filePath}");
                    return;
                }

                GameObject loadedObject = gltfRoot;
                Debug.Log($"GLB file loaded: {filePath}");
                PlaceObject(loadedObject, parentObject, generateRegion, position, scaleFactor);
            }
        );

        if (!loadFailed)
        {
            Debug.Log($"Successfully loaded GLB file: {filePath}");
        }

        filesLoaded++;
        Debug.Log($"Progress: {filesLoaded}/{filesToLoad} GLB files loaded.");

        if (filesLoaded >= filesToLoad)
        {
            IsLoadingComplete = true;
            Debug.Log("All GLB files have been loaded.");
        }
    }

    private static Vector3 GetRandomPosition(GameObject generateRegion, System.Random sysRandom)
    {
        // Generate a random position
        float xMin = generateRegion.transform.position.x - generateRegion.transform.localScale.x / 2;
        float xMax = generateRegion.transform.position.x + generateRegion.transform.localScale.x / 2;
        float zMin = generateRegion.transform.position.z - generateRegion.transform.localScale.z / 2;
        float zMax = generateRegion.transform.position.z + generateRegion.transform.localScale.z / 2;

        double randX = sysRandom.NextDouble() * (xMax - xMin) + xMin;
        double randZ = sysRandom.NextDouble() * (zMax - zMin) + zMin;
        float yPos = 0f; // Assuming ground level at y=0

        return new Vector3((float)randX, yPos, (float)randZ);
    }

    /// <summary>
    /// Places a loaded object randomly within the generate region without overlapping existing objects.
    /// </summary>
    private static void PlaceObject(GameObject obj, GameObject parentObject, GameObject generateRegion, Vector3 position, float scaleFactor)
    {
        obj.transform.SetParent(parentObject.transform);

        Vector3 genRegionSize = generateRegion.transform.localScale;
        Bounds objBounds = CalculateBounds(obj);
        // Scale the object to fit within the generate region
        float maxScale = Mathf.Min(
            genRegionSize.x / objBounds.size.x,
            genRegionSize.y / objBounds.size.y,
            genRegionSize.z / objBounds.size.z
        );

        float finalScale = maxScale * scaleFactor;
        obj.transform.localScale = new Vector3(finalScale, finalScale, finalScale);
        objBounds = CalculateBounds(obj);
        obj.transform.position = new Vector3(position.x, objBounds.extents.y, position.z);
        AddMeshColliders(obj);


        obj.name = "Obstacle";
        Rigidbody rb = obj.AddComponent<Rigidbody>();
        rb.constraints = RigidbodyConstraints.FreezePositionX | RigidbodyConstraints.FreezePositionZ | RigidbodyConstraints.FreezeRotation;
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
        rb.mass = 10000; // Set the mass of the object to 10000 to make it immovable
        rb.useGravity = true;
    }

    /// <summary>
    /// Adds MeshColliders to all MeshFilters in the GameObject and its children.
    /// </summary>
    private static void AddMeshColliders(GameObject obj)
    {
        MeshFilter[] meshFilters = obj.GetComponentsInChildren<MeshFilter>();
        foreach (MeshFilter meshFilter in meshFilters)
        {
            MeshCollider meshCollider = meshFilter.gameObject.AddComponent<MeshCollider>();
            meshCollider.sharedMesh = meshFilter.sharedMesh;
            meshCollider.convex = true; // Set to true if convex colliders are needed
        }
    }

    /// <summary>
    /// Deletes all child GameObjects of the specified parent.
    /// </summary>
    private static void DeleteAllChildren(GameObject parent)
    {
        // Iterate through all children of the parent GameObject
        for (int i = parent.transform.childCount - 1; i >= 0; i--)
        {
            GameObject child = parent.transform.GetChild(i).gameObject;
            Object.Destroy(child);
        }

        Debug.Log($"All children of {parent.name} have been deleted.");
    }

    /// <summary>
    /// Calculates the cumulative bounds of all renderers in the GameObject.
    /// </summary>
    private static Bounds CalculateBounds(GameObject obj)
    {
        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0)
        {
            return new Bounds(obj.transform.position, Vector3.zero);
        }

        Bounds bounds = renderers[0].bounds;
        foreach (Renderer renderer in renderers)
        {
            bounds.Encapsulate(renderer.bounds);
        }

        return bounds;
    }
}
