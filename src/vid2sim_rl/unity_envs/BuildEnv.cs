using UnityEngine;
using Unity.Collections;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections;
using GaussianSplatting.Runtime;
using UnityEngine.AI;

using Unity.Burst;
using Unity.Jobs;
using Unity.Mathematics;
using System;

public class CocoAgent : Agent
{   
    // Environment Assets
    public GameObject GSGameObject;
    public Camera MainCamera;
    public Camera DepthCamera;

    public GaussianSplatAsset GS;
    private NativeArray<GaussianSplatAsset.ChunkInfo> GSplat;

    Rigidbody rBody;
    public GameObject floor;
    public GameObject startRegion;
    public GameObject goalRegion; 

    private string folderPath = ""; // Your Folder containing GLB files
    public GameObject generateRegion;
    private GameObject parentObject;

    // Environment Parameters
    private EnvironmentParameters envParameters;
    public float timePenaltyMultiplier;
    public float distanceRewardMultiplier;
    public float collisionPenalty;
    public float maxEpisodeDuration;
    
    public float goalReward;
    public float failPenalty;

    // If the agent is within this distance of the target location, it will consider it reached the target.
    public float successRange = 0.5f;
    
    // Agent Settings
    public float moveMul = 1f;
    public float rotateMul = 0.5f;

    public bool enableRandomObjPlacement = false;
    public int maxObjNumber = 5;

    private Vector2 destination;
    private float episodeTime = 0f;
    private float previousDistance;
    private Vector3 floorNormal;
    private float[] imageArray;
    
    private bool collideFlag = false;
    private float collideCount = 0;
    private float stepCount = 0;
    public int randomSeed = 0;
    private System.Random systemRandom;

    // Agents
    private float maxSteeringAngle = 60f;
    private float wheelBase = 0.8f;

    // Cameras
    private float camPoseRandomRange;
    private float camRotRandomRange;
    public float perturbationFrequency;
    private Vector3 CameraInitialPosition;
    private Quaternion CameraInitialRotation;
    
    private bool isInference = false;
    
    private float lastMoveAction = -999f;
    private float lastSteerAction = -999f;
    private float steerSmoothRewardMultiplier = 0.1f;
    private int collisionLimit = 3;

    public override void Initialize()
    {
        base.Initialize();
        envParameters = Academy.Instance.EnvironmentParameters;
        timePenaltyMultiplier = envParameters.GetWithDefault("timePenaltyMultiplier", -0.1f);
        distanceRewardMultiplier = envParameters.GetWithDefault("distanceRewardMultiplier", 1f);
        steerSmoothRewardMultiplier = envParameters.GetWithDefault("steerSmoothRewardMultiplier", 0.05f);

        collisionPenalty = envParameters.GetWithDefault("collisionPenalty", -1f);
        goalReward = envParameters.GetWithDefault("goalReward", 10.0f);
        failPenalty = envParameters.GetWithDefault("failPenalty", -10.0f);

        maxEpisodeDuration = envParameters.GetWithDefault("maxEpisodeDuration", 60f);
        enableRandomObjPlacement = envParameters.GetWithDefault("enableRandomObjPlacement", 1f) > 0.5f;
        maxObjNumber = (int)envParameters.GetWithDefault("maxLoadedObjNum", 3f);
        collisionLimit = (int)envParameters.GetWithDefault("collisionLimit", 999f);
        
        moveMul = envParameters.GetWithDefault("moveMul", 1f);
        rotateMul = envParameters.GetWithDefault("rotateMul", 0.5f);
        randomSeed = (int)envParameters.GetWithDefault("randomSeed", 0f);
        isInference = envParameters.GetWithDefault("isInference", 0f) > 0.5f;
    }


    void Start()
    {
        parentObject = new GameObject("ObjParentObject");
        rBody = GetComponent<Rigidbody>();
        GSplat = GS.chunkData.GetData<GaussianSplatAsset.ChunkInfo>();
        floorNormal = floor.transform.up;
        rBody.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationY;
        rBody.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        rBody.interpolation = RigidbodyInterpolation.Interpolate;

        systemRandom = new System.Random(randomSeed);

        // Assume the MainCamera and DepthCamera are already set in the same position and rotation in the scene
        CameraInitialPosition = MainCamera.transform.localPosition;
        CameraInitialRotation = MainCamera.transform.localRotation;
    }

    void LateUpdate()
    {
        if(!isInference)
        {
            // Time variable to create smooth perturbations over time
            float time = Time.time * perturbationFrequency;
            float camPoseRandomRange_apply = camPoseRandomRange;
            float camRotRandomRange_apply = camRotRandomRange;

            // Apply smooth position perturbation (simulates small bumps)
            Vector3 positionPerturbation = new Vector3(
                Mathf.Sin(time * 0.5f) * camPoseRandomRange_apply,  // X oscillation
                Mathf.Sin(time * 0.7f) * camPoseRandomRange_apply,  // Y oscillation (different frequency for variation)
                Mathf.Sin(time * 0.6f) * camPoseRandomRange_apply   // Z oscillation
            );
            MainCamera.transform.localPosition = CameraInitialPosition + positionPerturbation;
            DepthCamera.transform.localPosition = CameraInitialPosition + positionPerturbation;

            // Apply smooth rotation perturbation (simulates small rotational bumps)
            Vector3 rotationPerturbation = new Vector3(
                Mathf.Sin(time * 0.3f) * camRotRandomRange_apply * 5,  // Roll oscillation
                Mathf.Sin(time * 0.4f) * camRotRandomRange_apply,      // Pitch oscillation
                Mathf.Sin(time * 0.35f) * camRotRandomRange_apply      // Yaw oscillation
            );
            MainCamera.transform.localRotation = Quaternion.Euler(CameraInitialRotation.eulerAngles + rotationPerturbation);
            DepthCamera.transform.localRotation = Quaternion.Euler(CameraInitialRotation.eulerAngles + rotationPerturbation);
        }
    }

    public override void OnEpisodeBegin()
    {
        // Start a coroutine to handle initialization sequentially
        StartCoroutine(InitializeEpisode());
    }

    private IEnumerator InitializeEpisode()
    {
        Vector3 startRegionLoc = startRegion.transform.position;
        Vector3 startRegionSize = startRegion.transform.localScale;

        float agentHeight = GetComponent<Collider>().bounds.size.y;

        Vector3 agentPosition = GetRandomPosition(startRegionLoc, startRegionSize);
        transform.position = agentPosition;

        float randomRotationZ = GetRandomFloat(-30f, 30f);
        transform.rotation = Quaternion.Euler(90f, 0f, 180f - randomRotationZ);
        rBody.velocity = Vector3.zero;

        Vector3 goalRegionLoc = goalRegion.transform.position;
        Vector3 goalRegionSize = goalRegion.transform.localScale;

        float goalX = GetRandomFloat(goalRegionLoc.x - goalRegionSize.x / 2, goalRegionLoc.x + goalRegionSize.x / 2);
        float goalZ = GetRandomFloat(goalRegionLoc.z - goalRegionSize.z / 2, goalRegionLoc.z + goalRegionSize.z / 2);
        destination = new Vector2(goalX, goalZ);

        previousDistance = CalculateDistanceToGoal();
        
        episodeTime = 0f;
        stepCount = 0;
        collideCount = 0f;
        lastMoveAction = -999f;
        lastSteerAction = -999f;
        collideFlag = false;
        
        rBody.angularVelocity = Vector3.zero;
        rBody.velocity = Vector3.zero;
        rBody.angularDrag = 0f; 
        rBody.drag = 0f; 

        camPoseRandomRange = GetRandomFloat(0.0f, 0.02f);
        camRotRandomRange = GetRandomFloat(0.0f, 0.3f);
        perturbationFrequency = GetRandomFloat(2.0f, 10.0f);

        // If random object placement is enabled, load and place the assets
        if (enableRandomObjPlacement)
        {
            GLBLoader.ResetLoadingComplete(); // Reset the loading complete flag
            GLBLoader.LoadAndPlaceAssets(maxObjNumber, folderPath, parentObject, generateRegion, this, systemRandom);
            // Wait until all GLB files are loaded
            while (!GLBLoader.IsLoadingComplete)
            {
                yield return null;
            }
        }
    }

    private float GetRandomFloat(float min, float max)
    {
        return (float)(systemRandom.NextDouble() * (max - min) + min);
    }

    private Vector3 GetRandomPosition(Vector3 center, Vector3 size)
    {
        float x = GetRandomFloat(center.x - size.x / 2, center.x + size.x / 2);
        float z = GetRandomFloat(center.z - size.z / 2, center.z + size.z / 2);
        return new Vector3(x, 0.02f, z);
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.name == "Obstacle")
        {
            collideFlag = true;
            Debug.Log("Collision with mesh object detected, applying penalty!");
        }
    }

    private void HandleCollision()
    {
        if (collideFlag)
        {
            collideCount++;
            AddReward(collisionPenalty);
            collideFlag = false;
        }
    }


    private void ApplyTimePenalty()
    {
        float timePenalty = timePenaltyMultiplier * Time.deltaTime;
        AddReward(timePenalty);
        episodeTime += Time.deltaTime;
    }


    private float CalculateDistanceToGoal()
    {
        float xPos = transform.position.x;
        float zPos = transform.position.z;
        return Vector2.Distance(new Vector2(xPos, zPos), destination);
    }

    private void CheckFail()
    {
        float yPos = transform.position.y;
        if(yPos < -2){ // -1 is a heuristics value 
            AddReward(failPenalty); // Fail, add a large negative reward
            EndEpisode();
        }
        if (collideCount >= collisionLimit)
        {
            AddReward(failPenalty); // Fail, add a large negative reward
            EndEpisode();
        }
    }

    private void CheckSuccess()
    {
        float currentDistance = CalculateDistanceToGoal();
        float deltaDistance = previousDistance - currentDistance;
        float reward = deltaDistance * distanceRewardMultiplier;
        AddReward(reward);
        previousDistance = currentDistance;
        if (currentDistance < successRange)
        {
            AddReward(goalReward);
            EndEpisode();
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {       
        // Info channel, do not treat as observation
        float collisionRate = 0;
        if(stepCount == 0){
            collisionRate = -1;
        } else {
            collisionRate = collideCount / stepCount;
        }
        sensor.AddObservation(collideCount);
        sensor.AddObservation(collisionRate);

        Vector2 toDestination = new Vector2(destination.x, destination.y) - new Vector2(transform.position.x, transform.position.z);
        Vector2 forward = new Vector2(transform.up.x, transform.up.z);
        float angleToDestination = Vector2.SignedAngle(forward, toDestination);
        float normalizedAngle = angleToDestination / 180.0f;

        sensor.AddObservation(normalizedAngle);
        sensor.AddObservation(toDestination.magnitude);
    }

    public void Act(ActionBuffers actionBuffers)
    {
        // Retrieve continuous actions
        float moveInput = Mathf.Clamp(actionBuffers.ContinuousActions[0], 0f, 1f);    // Forward Movement
        float steerInput = Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f);   // Steering

        // Parameters
        float speed = moveInput * moveMul;                        // Linear speed
        
        float steeringAngle = steerInput * maxSteeringAngle;      // Current steering angle in degrees
        float dt = Time.deltaTime;                                // Time step

        // Convert angles to radians for calculations
        float deltaRad = steeringAngle * Mathf.Deg2Rad;
        // Update orientation (heading)
        float angularVelocity = (speed / wheelBase) * Mathf.Tan(deltaRad) * rotateMul;
        float newThetaRad = -angularVelocity * dt;

        // Update position
        float dx = speed * Mathf.Sin(newThetaRad) * dt;
        float dy = -speed * Mathf.Cos(newThetaRad) * dt;
        Vector3 controlSignal = new Vector3(dx, dy, 0);
        Vector3 newPosition = rBody.position + transform.TransformDirection(controlSignal);
        Quaternion newRotation = rBody.rotation * Quaternion.Euler(0f, 0f, newThetaRad * Mathf.Rad2Deg);

        // Apply new position and rotation
        rBody.MovePosition(newPosition);
        rBody.MoveRotation(newRotation);
    }

    public void ApplyActionReward(ActionBuffers actionBuffers)
    {
        float moveInput = Mathf.Clamp(actionBuffers.ContinuousActions[0], 0f, 1f);    // Forward Movement
        float steerInput = Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f);   // Steering

        if (lastMoveAction >= -1f && lastSteerAction >= -1)
        {
            float speed = moveInput * moveMul;
            float steerDiff = Mathf.Abs(steerInput - lastSteerAction);
            float reward = -steerDiff * speed * steerSmoothRewardMultiplier;
            AddReward(reward);
        } 
        
        lastMoveAction = moveInput;
        lastSteerAction = steerInput;
    }


    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        stepCount++;
        Act(actionBuffers);
        ApplyActionReward(actionBuffers);
        
        HandleCollision();
        CheckSuccess();
        CheckFail();
        ApplyTimePenalty();
        if (episodeTime >= maxEpisodeDuration){
            EndEpisode();
        }
        Debug.Log("Reward: " + GetCumulativeReward());
    }

    void OnDestroy()
    {
        if (GSplat.IsCreated)
            GSplat.Dispose();
        
        if (parentObject != null)
            UnityEngine.Object.Destroy(parentObject);
    }
    
    // Heuristic method
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxis("Vertical");
        continuousActions[1] = Input.GetAxis("Horizontal");
    }
}
