using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class Chaser : MonoBehaviour
{
    private Transform agent;
    private Rigidbody rb;
    public float speed = 80f;
    public float maxSpeed = 80f;

    public void Start()
    {
        rb = GetComponent<Rigidbody>();
        agent = GameObject.FindGameObjectWithTag("Agent").transform;
    }

    private void FixedUpdate()
    {
        Vector3 velocity = agent.position - transform.position;
        rb.AddForce(velocity.normalized * speed * Time.fixedDeltaTime, ForceMode.VelocityChange);
    }
    public void IncrementSpeed()
    {
        speed += 0.016f;
        speed = Mathf.Clamp(speed, 0f, maxSpeed);
    }
}
