using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.VisualScripting;
using UnityEngine;

public class ChaseAgent : Agent
{
    public float totalAngle = 360;
    public float numberRays = 32;
    public float speed = 0.1f;
    public int bound = 100;
    private float delta;
    private RaycastHit hit;
    private CapsuleCollider capsuleCollider;
    private Rigidbody rb;
    private Chaser[] chasers;
    private GameObject target;

    protected override void Awake()
    {
        base.Awake();
        delta = totalAngle / numberRays;
        capsuleCollider = GetComponent<CapsuleCollider>();
        rb = GetComponent<Rigidbody>();
        chasers = FindObjectsByType<Chaser>(FindObjectsSortMode.None);
        target = GameObject.FindGameObjectWithTag("Target");
    }

    public override void OnEpisodeBegin()
    {
        Vector3 startPos = new(Random.Range(-bound * 2, bound * 2) + target.transform.position.x, capsuleCollider.height / 2, Random.Range(-bound * 2, bound * 2) + target.transform.position.z);
        List<Vector3> chasersPos = new();

        for (int i = 0; i < chasers.Length; i++)
        {
            Vector3 tempPos = new(Random.Range(-bound, bound) + target.transform.position.x, capsuleCollider.height / 2, Random.Range(-bound, bound) + target.transform.position.z);
            float dist = Vector3.Distance(tempPos, startPos);
            while (dist < 5)
            {
                tempPos = new(Random.Range(-bound, bound) + target.transform.position.x, capsuleCollider.height / 2, Random.Range(-bound, bound) + target.transform.position.z);
                dist = Vector3.Distance(tempPos, startPos);
            }

            chasersPos.Add(tempPos);
            chasers[i].IncrementSpeed();
        }
        for (int i = 0; i < chasers.Length; i++)
        {
            chasers[i].transform.position = chasersPos[i];
        }
        chasersPos.Clear();

        rb.linearVelocity = Vector3.zero;

        transform.position = startPos;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        foreach (Chaser chaser in chasers)
        {
            sensor.AddObservation(chaser.transform.position - transform.position);
        }
        //sensor.AddObservation(transform.position);

        sensor.AddObservation(target.transform.position - transform.position);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Action
        float vertical = actionBuffers.ContinuousActions[0];
        float horizontal = actionBuffers.ContinuousActions[1];
        Vector3 velocity = new(vertical, 0, horizontal);

        float minDist = 100f;
        foreach (Chaser chaser in chasers)
        {
            float tempDist = Vector3.Distance(transform.position, chaser.transform.position);
            if (tempDist < minDist)
                minDist = tempDist;
        }

        rb.AddForce(velocity.normalized * speed * Time.fixedDeltaTime, ForceMode.VelocityChange);

        // Reward
        AddReward(-1e-4f * Vector3.Distance(transform.position, target.transform.position));
        foreach (Chaser chaser in chasers)
        {
            float tempDist = Vector3.Distance(transform.position, chaser.transform.position);
            AddReward(-0.003f * Mathf.Exp(-(tempDist - 2)));
        }

        if (minDist < 1)
        {
            SetReward(-10);
            Debug.Log("Failed");
            EndEpisode();
        } 
        else if (Vector3.Distance(transform.position, target.transform.position) < 2)
        {
            SetReward(100);
            Debug.Log("Succeeded");
            EndEpisode();
        }
        
    }
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Vertical");
        continuousActionsOut[1] = Input.GetAxis("Horizontal");
    }
}
