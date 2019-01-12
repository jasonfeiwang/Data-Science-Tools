import boto
import boto.s3
import sys
from boto.s3.key import Key

class S3():

    def upload(self, file):
        AWS_ACCESS_KEY_ID = 'key'
        AWS_SECRET_ACCESS_KEY = 'secret'

        bucket_name = AWS_ACCESS_KEY_ID.lower() + '-dump'
        conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
                               AWS_SECRET_ACCESS_KEY)

        bucket = conn.create_bucket(bucket_name,location=boto.s3.connection.Location.DEFAULT)

        print 'Uploading %s to Amazon S3 bucket %s' % (file, bucket_name)

        def percent_cb(complete, total):
            sys.stdout.write('.')
            sys.stdout.flush()

        k = Key(bucket)
        k.key = file
        k.set_contents_from_filename(file, cb=percent_cb, num_cb=10)
